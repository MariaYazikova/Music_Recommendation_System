import torch
import tqdm
import json
import numpy as np
import pandas as pd
import time
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# =========================
# DATA
# =========================

X_train = pd.read_pickle("X_train.pkl")
X_val = pd.read_pickle("X_val.pkl")
X_test = pd.read_pickle("X_test.pkl")

all_tracks = pd.concat([X_train, X_val, X_test])

id_to_pos = {track_id: pos for pos, track_id in enumerate(all_tracks.index)}
pos_to_id = {pos: track_id for track_id, pos in id_to_pos.items()}


# =========================
# LOAD KNN CONFIGS
# =========================

def load_knn_config(model_type="audio"):
    if model_type == "audio":
        cfg_path = "best_knn_audio_config.pkl"
    else:
        cfg_path = "best_knn_hybrid_config.pkl"

    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    return config


# =========================
# METRICS
# =========================

def average_precision(rels, m, N):
    if m == 0:
        return 0

    hits = 0
    ap_sum = 0

    for k, r in enumerate(rels, 1):
        if r:
            hits += 1
            ap_sum += hits / k

    return ap_sum / min(m, N)


def average_recall(rels, m, N):
    if m == 0:
        return 0

    hits = 0
    ar_sum = 0

    for k, r in enumerate(rels, 1):
        if r:
            hits += 1
            ar_sum += hits / m

    return ar_sum / min(m, N)


def reciprocal_rank(rels):
    for i, r in enumerate(rels, 1):
        if r:
            return 1 / i
    return 0


def diversity(embeddings):
    k = len(embeddings)
    if k < 2:
        return 0

    sim = cosine_similarity(embeddings)

    total = 0
    count = 0

    for i in range(k):
        for j in range(i + 1, k):
            total += sim[i][j]
            count += 1

    return 1 - total / count


# =========================
# TRACK-TO-TRACK (COSINE)
# =========================

def evaluate_track_based_cosine(embeddings_np, test_embeddings_np, genre_matrix_np, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    test_embeddings = torch.from_numpy(test_embeddings_np).float().to(device)
    genres = torch.from_numpy(genre_matrix_np).float().to(device)

    AP10, AR10, MRR10 = [], [], []
    AP50, AR50, MRR50 = [], [], []
    div10, div50 = [], []

    all_recommended = set()

    n_tracks = embeddings.shape[0]

    for i in tqdm.tqdm(range(len(test_embeddings))):

        sims = torch.mm(test_embeddings[i].unsqueeze(0), embeddings.T)[0]

        mask = torch.zeros(n_tracks, device=device)
        mask[i] = -1e9
        sims = sims + mask

        rec_vals, rec_pos = torch.topk(sims, k=k)

        rec_pos_np = rec_pos.cpu().numpy()

        rec_genres = genres[rec_pos]
        liked_genres = genres[i].unsqueeze(0)

        rels = (rec_genres * liked_genres.unsqueeze(0)).any(dim=1).int().cpu().numpy()

        overlaps = torch.mm(genres, liked_genres.T)
        m = (overlaps > 0).sum().item()

        rec_embs = embeddings_np[rec_pos_np]

        AP10.append(average_precision(rels[:10], m, 10))
        AR10.append(average_recall(rels[:10], m, 10))
        MRR10.append(reciprocal_rank(rels[:10]))

        AP50.append(average_precision(rels, m, 50))
        AR50.append(average_recall(rels, m, 50))
        MRR50.append(reciprocal_rank(rels))

        div10.append(diversity(rec_embs[:10]))
        div50.append(diversity(rec_embs))

        all_recommended.update(rec_pos_np)

    return {
        "MAP@10": float(np.mean(AP10)),
        "MAP@50": float(np.mean(AP50)),
        "MAR@10": float(np.mean(AR10)),
        "MAR@50": float(np.mean(AR50)),
        "MRR@10": float(np.mean(MRR10)),
        "MRR@50": float(np.mean(MRR50)),
        "Coverage": float(len(all_recommended) / n_tracks),
        "Diversity@10": float(np.mean(div10)),
        "Diversity@50": float(np.mean(div50)),
    }


# =========================
# TRACK-TO-TRACK (KNN)
# =========================

def evaluate_track_based_knn(embeddings_np, test_embeddings_np, genre_matrix_np, k=10, model_type="audio"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_knn_config(model_type)


    knn = NearestNeighbors(
        n_neighbors=k,
        metric=config.get("metric", "cosine"),
        algorithm=config.get("algorithm", "auto")
    )

    knn.fit(embeddings_np)

    genres = torch.from_numpy(genre_matrix_np).float().to(device)

    AP10, AR10, MRR10 = [], [], []
    AP50, AR50, MRR50 = [], [], []
    div10, div50 = [], []

    all_recommended = set()

    n_tracks = embeddings_np.shape[0]

    for i in tqdm.tqdm(range(len(test_embeddings_np))):

        dist, idx = knn.kneighbors(test_embeddings_np[i].reshape(1, -1), n_neighbors=k)

        rec_pos = idx[0]

        rec_genres = genres[rec_pos]
        liked_genres = genres[i]

        rels = (rec_genres * liked_genres.unsqueeze(0)).any(dim=1).int().cpu().numpy()

        m = ((genres * liked_genres).sum(dim=1) > 0).sum().item()

        rec_embs = embeddings_np[rec_pos]

        AP10.append(average_precision(rels[:10], m, 10))
        AR10.append(average_recall(rels[:10], m, 10))
        MRR10.append(reciprocal_rank(rels[:10]))

        AP50.append(average_precision(rels, m, 50))
        AR50.append(average_recall(rels, m, 50))
        MRR50.append(reciprocal_rank(rels))

        div10.append(diversity(rec_embs[:10]))
        div50.append(diversity(rec_embs))

        all_recommended.update(rec_pos)

    return {
        "MAP@10": float(np.mean(AP10)),
        "MAP@50": float(np.mean(AP50)),
        "MAR@10": float(np.mean(AR10)),
        "MAR@50": float(np.mean(AR50)),
        "MRR@10": float(np.mean(MRR10)),
        "MRR@50": float(np.mean(MRR50)),
        "Coverage": float(len(all_recommended) / n_tracks),
        "Diversity@10": float(np.mean(div10)),
        "Diversity@50": float(np.mean(div50)),
    }


# =========================
# USER-TO-TRACK (COSINE)
# =========================

def evaluate_user_based_cosine(user_liked_lists, embeddings_np, genre_matrix_np, k=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    genres = torch.from_numpy(genre_matrix_np).float().to(device)

    AP10, AR10, MRR10 = [], [], []
    AP50, AR50, MRR50 = [], [], []
    div10, div50 = [], []

    all_recommended = set()

    n_tracks = embeddings.shape[0]

    for user in tqdm.tqdm(user_liked_lists):

        user_emb = user["vector"]

        sims = torch.mm(user_emb, embeddings.T)[0]

        liked_pos = torch.tensor(
            [id_to_pos[x] for x in user["liked_tracks"]],
            device=device
        )

        mask = torch.zeros(n_tracks, device=device)
        mask[liked_pos] = -1e9
        sims = sims + mask

        rec_vals, rec_pos = torch.topk(sims, k=k)

        rec_pos_np = rec_pos.cpu().numpy()

        rec_genres = genres[rec_pos]
        liked_genres = genres[liked_pos]

        user_genre_mask = liked_genres.any(dim=0)

        rels = (rec_genres * user_genre_mask.unsqueeze(0)).any(dim=1).int().cpu().numpy()

        m = ((genres * user_genre_mask).sum(dim=1) > 0).sum().item()

        rec_embs = embeddings_np[rec_pos_np]

        AP10.append(average_precision(rels[:10], m, 10))
        AR10.append(average_recall(rels[:10], m, 10))
        MRR10.append(reciprocal_rank(rels[:10]))

        AP50.append(average_precision(rels, m, 50))
        AR50.append(average_recall(rels, m, 50))
        MRR50.append(reciprocal_rank(rels))

        div10.append(diversity(rec_embs[:10]))
        div50.append(diversity(rec_embs))

        all_recommended.update(rec_pos_np)

    return {
        "MAP@10": float(np.mean(AP10)),
        "MAP@50": float(np.mean(AP50)),
        "MAR@10": float(np.mean(AR10)),
        "MAR@50": float(np.mean(AR50)),
        "MRR@10": float(np.mean(MRR10)),
        "MRR@50": float(np.mean(MRR50)),
        "Coverage": float(len(all_recommended) / n_tracks),
        "Diversity@10": float(np.mean(div10)),
        "Diversity@50": float(np.mean(div50)),
    }


# =========================
# USER-TO-TRACK (KNN)
# =========================

def evaluate_user_based_knn(user_liked_lists, embeddings_np, genre_matrix_np, k=10, model_type="audio"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_knn_config(model_type)

    knn = NearestNeighbors(
        n_neighbors=k,
        metric=config.get("metric", "cosine"),
        algorithm=config.get("algorithm", "auto")
    )

    knn.fit(embeddings_np)

    genres = torch.from_numpy(genre_matrix_np).float().to(device)

    AP10, AR10, MRR10 = [], [], []
    AP50, AR50, MRR50 = [], [], []
    div10, div50 = [], []

    all_recommended = set()

    n_tracks = embeddings_np.shape[0]
    inference_times = []

    for user in tqdm.tqdm(user_liked_lists):
        
        user_emb = user["vector"]

        start = time.time()

        dist, idx = knn.kneighbors(user_emb, n_neighbors=k)

        end = time.time()
        inference_times.append(end - start)

        rec_pos = idx[0]

        liked_pos = torch.tensor(
            [id_to_pos[x] for x in user["liked_tracks"]],
            device=device
        )

        rec_genres = genres[rec_pos]
        liked_genres = genres[liked_pos]

        user_genre_mask = liked_genres.any(dim=0)

        rels = (rec_genres * user_genre_mask.unsqueeze(0)).any(dim=1).int().cpu().numpy()

        m = ((genres * user_genre_mask).sum(dim=1) > 0).sum().item()

        rec_embs = embeddings_np[rec_pos]

        AP10.append(average_precision(rels[:10], m, 10))
        AR10.append(average_recall(rels[:10], m, 10))
        MRR10.append(reciprocal_rank(rels[:10]))

        AP50.append(average_precision(rels, m, 50))
        AR50.append(average_recall(rels, m, 50))
        MRR50.append(reciprocal_rank(rels))

        div10.append(diversity(rec_embs[:10]))
        div50.append(diversity(rec_embs))

        all_recommended.update(rec_pos)

    return {
        "MAP@10": float(np.mean(AP10)),
        "MAP@50": float(np.mean(AP50)),
        "MAR@10": float(np.mean(AR10)),
        "MAR@50": float(np.mean(AR50)),
        "MRR@10": float(np.mean(MRR10)),
        "MRR@50": float(np.mean(MRR50)),
        "Coverage": float(len(all_recommended) / n_tracks),
        "Diversity@10": float(np.mean(div10)),
        "Diversity@50": float(np.mean(div50)),
        "Mean inference time": float(np.mean(inference_times)),
    }


# =========================
# METRICS STORAGE
# =========================

ALL_METRICS = {}

def add_model_metrics(model_name, metrics_dict, filename="all_model_metrics.json"):

    global ALL_METRICS
    ALL_METRICS[model_name] = metrics_dict

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ALL_METRICS, f, indent=4, ensure_ascii=False)

    print(f"Saved metrics for {model_name}")