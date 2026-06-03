import pickle
import numpy as np
import pandas as pd

users_url = '/dataset/users.pkl'
with open(users_url, 'rb') as c:
    users = pickle.load(c)

all_tracks_embeddings = np.load('/embeddings/all_embeddings.npy')
test_tracks_embeddings = np.load('/embeddings/test_embeddings.npy')

all_tracks_hybrid_embeddings = np.load('/embeddings/all_embeddings_hybrid.npy')
test_tracks_hybrid_embeddings = np.load('/embeddings/test_embeddings_hybrid.npy')

ae_embeddings_np = np.load('/embeddings/ae_audio_embeddings.npy')
test_ae_embeddings_np = np.load('/embeddings/ae_audio_test_embeddings.npy')

ae_hybrid_embeddings_np = np.load('/embeddings/ae_hybrid_embeddings.npy')
test_ae_hybrid_embeddings_np = np.load('/embeddings/ae_hybrid_test_embeddings.npy')

with open('/embeddings/id_to_pos.pkl', 'rb') as f:
    id_to_pos = pickle.load(f)

with open('/embeddings/pos_to_id.pkl', 'rb') as g:
    pos_to_id = pickle.load(g)

BASE_PATH = '/dataset/'

train_url = f'{BASE_PATH}X_train.pkl'
with open(train_url, 'rb') as a:
    train_df = pickle.load(a)

val_url = f'{BASE_PATH}X_val.pkl'
with open(val_url, 'rb') as b:
    val_df = pickle.load(b)

test_url = f'{BASE_PATH}X_test.pkl'
with open(test_url, 'rb') as c:
    test_df = pickle.load(c)

all_tracks = pd.concat([train_df, val_df, test_df])
import torch
import tqdm
import json
import numpy as np
import pandas as pd
import time
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

id_to_pos = {track_id: pos for pos, track_id in enumerate(all_tracks.index)}
pos_to_id = {pos: track_id for track_id, pos in id_to_pos.items()}


# Загрузка KNN конфигураций
def load_knn_config(model_type="audio"):
    if model_type == "audio":
        cfg_path = "/embeddings/best_knn_audio_config.pkl"
    else:
        cfg_path = "/embeddings/best_knn_hybrid_config.pkl"

    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    return config


# Метрики
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


# Track-To-Track (Cosine)
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
        rels = rels.flatten().tolist()

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


# Track-To-Track (KNN)
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
        if isinstance(test_embeddings_np, torch.Tensor):
            test_embeddings_np = test_embeddings_np.cpu().numpy()
        if test_embeddings_np.ndim == 1:
            test_embeddings_np = test_embeddings_np.reshape(1, -1)

        dist, idx = knn.kneighbors(test_embeddings_np[i].reshape(1, -1), n_neighbors=k)

        rec_pos = idx[0]

        rec_genres = genres[rec_pos]
        liked_genres = genres[i]

        rels = (rec_genres * liked_genres.unsqueeze(0)).any(dim=1).int().cpu().numpy()
        rels = rels.flatten().tolist()

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


# User-To-Track (Cosine)
def evaluate_user_based_cosine(user_liked_lists, embeddings_np, genre_matrix_np, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    genres = torch.from_numpy(genre_matrix_np).float().to(device)

    AP10, AR10, MRR10 = [], [], []
    AP50, AR50, MRR50 = [], [], []
    div10, div50 = [], []

    all_recommended = set()

    n_tracks = embeddings.shape[0]
    inference_times = []

    for user in tqdm.tqdm(user_liked_lists):
        user_emb = user["vector"]
        start = time.time()

        sims = torch.mm(user_emb, embeddings.T)[0]

        liked_pos = torch.tensor(
            [id_to_pos[x] for x in user["liked_tracks"]],
            device=device
        )

        mask = torch.zeros(n_tracks, device=device)
        mask[liked_pos] = -1e9
        sims = sims + mask

        rec_vals, rec_pos = torch.topk(sims, k=k)
        end = time.time()
        inference_times.append(end - start)

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
        "Mean inference time": float(np.mean(inference_times)),

    }


# User-To-Track (KNN)
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
        if isinstance(user_emb, torch.Tensor):
            user_emb = user_emb.cpu().numpy()

        if user_emb.ndim == 1:
            user_emb = user_emb.reshape(1, -1)

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


# Расчёт метрик
ALL_METRICS = {}


def add_model_metrics(model_name, metrics_dict, filename="all_model_metrics.json"):
    global ALL_METRICS
    ALL_METRICS[model_name] = metrics_dict

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ALL_METRICS, f, indent=4, ensure_ascii=False)

    print(f"Saved metrics for {model_name}")


import torch


def evaluate_metrics_graph(num_features, test_tracks_embeddings):
    import tqdm
    def evaluate_user_vector_graph(user_liked_lists, features_np, embeddings_np, k):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = torch.from_numpy(features_np).float().to(device)
        embeddings = torch.from_numpy(embeddings_np).float().to(device)

        for user in tqdm.tqdm(user_liked_lists):
            favorite_poses = [id_to_pos[track_id] for track_id in user['liked_tracks']]
            liked_poses_tensor = torch.tensor(favorite_poses, dtype=torch.long, device=device)
            favorite_vectors = features[liked_poses_tensor]
            mean_vec_raw = favorite_vectors.mean(dim=0, keepdim=True)

            sims_neighbors = torch.mm(mean_vec_raw, features.T)[0]

            top_k_scores, top_k_poses = torch.topk(sims_neighbors, k)

            weights = top_k_scores.unsqueeze(1)
            neighbours_emb = embeddings[top_k_poses]

            new_emb = (neighbours_emb * weights).sum(dim=0) / weights.sum()
            new_emb = new_emb.reshape(1, -1)

            user['vector'] = new_emb

        return user_liked_lists

    features_np = all_tracks.iloc[:, :num_features].values
    user_liked_list = evaluate_user_vector_graph(users, features_np, all_tracks_embeddings, 10)

    metrics_user_cosine = evaluate_user_based_cosine(user_liked_list, all_tracks_embeddings, genre_matrix_np, 50)

    for metric in metrics_user_cosine.keys():
        print(f"Metric: {metric} = {metrics_user_cosine[metric]:.8f}")

    if num_features == 518:
        metrics_user_knn = evaluate_user_based_knn(user_liked_list, all_tracks_embeddings, genre_matrix_np, 50, 'audio')
    else:
        metrics_user_knn = evaluate_user_based_knn(user_liked_list, all_tracks_embeddings, genre_matrix_np, 50,
                                                   'hybrid')

    for metric in metrics_user_knn.keys():
        print(f"Metric: {metric} = {metrics_user_knn[metric]:.8f}")

    metrics_track_cosine = evaluate_track_based_cosine(all_tracks_embeddings, test_tracks_embeddings, genre_matrix_np,
                                                       50)

    for metric in metrics_track_cosine.keys():
        print(f"Metric: {metric} = {metrics_track_cosine[metric]:.8f}")

    if num_features == 518:
        metrics_track_knn = evaluate_track_based_knn(all_tracks_embeddings, test_tracks_embeddings, genre_matrix_np, 50,
                                                     'audio')
    else:
        metrics_track_knn = evaluate_track_based_knn(all_tracks_embeddings, test_tracks_embeddings, genre_matrix_np, 50,
                                                     'hybrid')
    for metric in metrics_track_knn.keys():
        print(f"Metric: {metric} = {metrics_track_knn[metric]:.8f}")

    if num_features == 518:
        model_name = f"GNN_Audio"
    else:
        model_name = f"GNN_Hybrid"
    combined_metrics = {
        "User_to_Track_Cosine": metrics_user_cosine,
        "User_to_Track_Knn": metrics_user_knn,
        "Track_to_Track_Cosine": metrics_track_cosine,
        "Track_to_Track_Knn": metrics_track_knn
    }

    add_model_metrics(model_name, combined_metrics)


from torch.utils.data import TensorDataset, DataLoader


def evaluate_metrics_ae(num_features, test_embeddings_np):
    def evaluate_user_vectors_ae(users, embeddings_np):
        for user in tqdm.tqdm(users):
            favorite_indices = [id_to_pos[track_id] for track_id in user['liked_tracks']]
            favorite_vectors = embeddings_np[favorite_indices]
            mean_vector = np.mean(favorite_vectors, axis=0)
            user_emb = torch.from_numpy(mean_vector).float().to(device)
            if user_emb.dim() == 1:
                user_emb = user_emb.unsqueeze(0)
            user['vector'] = user_emb
        return users

    import torch
    features_np = all_tracks.iloc[:, :num_features].values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    user_liked_lists = evaluate_user_vectors_ae(users, ae_embeddings_np)
    genre_matrix_np = all_tracks[all_tracks.columns[518:535]].values.astype(np.float32)

    metrics_user_cosine = evaluate_user_based_cosine(user_liked_lists, ae_embeddings_np, genre_matrix_np)
    for metric in metrics_user_cosine.keys():
        print(f"Metric: {metric} = {round(metrics_user_cosine[metric], 8)}")

    if num_features == 518:
        metrics_user_knn = evaluate_user_based_knn(user_liked_lists, ae_embeddings_np, genre_matrix_np, 50, 'audio')
    else:
        metrics_user_knn = evaluate_user_based_knn(user_liked_lists, ae_embeddings_np, genre_matrix_np, 50, 'hybrid')

    for metric in metrics_user_knn.keys():
        print(f"Metric: {metric} = {round(metrics_user_knn[metric], 8)}")

    metrics_track_cosine = evaluate_track_based_cosine(ae_embeddings_np, test_embeddings_np, genre_matrix_np)
    for metric in metrics_track_cosine.keys():
        print(f"Metric: {metric} = {round(metrics_track_cosine[metric], 8)}")

    if num_features == 518:
        metrics_track_knn = evaluate_track_based_knn(ae_embeddings_np, test_embeddings_np, genre_matrix_np, 50, 'audio')
    else:
        metrics_track_knn = evaluate_track_based_knn(ae_embeddings_np, test_embeddings_np, genre_matrix_np, 50,
                                                     'hybrid')

    for metric in metrics_track_knn.keys():
        print(f"Metric: {metric} = {round(metrics_track_knn[metric], 8)}")

    combined_metrics = {
        "User_to_Track_Cosine": metrics_user_cosine,
        "User_to_Track_Knn": metrics_user_knn,
        "Track_to_Track_Cosine": metrics_track_cosine,
        "Track_to_Track_Knn": metrics_track_knn
    }
    if num_features == 518:
        model_name = f"Autoencoder_Audio"
    else:
        model_name = f"Autoencoder_Hybrid"

    add_model_metrics(model_name, combined_metrics)


genre_cols = all_tracks.columns[518:535]

genre_matrix_np = pd.DataFrame(
    data=all_tracks[genre_cols].values.astype(np.float32),
    columns=genre_cols,
    index=all_tracks.index
)
genre_matrix_np.to_csv('genre_matrix_np.csv', index=True)
import json
import tqdm
import time

genre_matrix_np = all_tracks[all_tracks.columns[518:535]].values.astype(np.float32)

all_metrics = {}
evaluate_metrics_ae(518, test_ae_embeddings_np)
evaluate_metrics_ae(535, test_ae_embeddings_np)
evaluate_metrics_graph(518, test_tracks_embeddings)
evaluate_metrics_graph(535, test_tracks_hybrid_embeddings)
