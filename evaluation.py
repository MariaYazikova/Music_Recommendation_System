import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# релевантность track-to-track
# имеют ли два трека i j хотя бы один общий жанр
def is_relevant(i, j, genre_matrix):
    return np.dot(
        genre_matrix.iloc[i].values,
        genre_matrix.iloc[j].values
    ) > 0

# релевантность user-to-track
# есть ли у рекомендуемого трека хотя бы один
# общий жанр с любым из сохраненных треков пользователем
def is_relevant_user(candidate_idx, liked_indices, genre_matrix):

    c = genre_matrix.iloc[candidate_idx].values

    for i in liked_indices:
        if np.dot(c, genre_matrix.iloc[i].values) > 0:
            return True

    return False

# индкесы k наиболее похожих треков
def get_top_k(sim_row, k):
    return np.argsort(sim_row)[::-1][1:k+1]

# AP@N
def average_precision(rels, m, N):
    # m - кол-во релевантных треков для данного объекта во всем датасете 
    # N - кол-во рекомендаций (top-N)
    if m == 0:
        return 0
    
    # hits - сколько релевантных треков найдено до позиции k
    hits = 0
    # ap_sum - сумма Presision для релевантных треков
    ap_sum = 0

    for k, r in enumerate(rels[:N], 1):

        if r:
            hits += 1
            p_k = hits / k
            ap_sum += p_k

    return ap_sum / min(m, N)

# AR@N
def average_recall(rels, m, N):

    if m == 0:
        return 0

    hits = 0
    ar_sum = 0

    for k, r in enumerate(rels[:N], 1):

        if r:
            hits += 1
            r_k = hits / m
            ar_sum += r_k

    return ar_sum / min(m, N)

# RR
def reciprocal_rank(rels):

    for i, r in enumerate(rels, 1):
        # i - позиция первого релевантного результата
        if r:
            return 1 / i

    return 0

# Diversity
def diversity(embeddings):

    k = len(embeddings)
    if k < 2:
        return 0

    # матрица сходства между всеми парами элементов
    sim = cosine_similarity(embeddings)

    # total - сумма попарных сходств между всеми элементами
    total = 0
    # count - кол-во всех уникальных пар (=K(K-1)/2)
    count = 0

    for i in range(k):
        for j in range(i + 1, k):
            total += sim[i][j]
            count += 1

    ils = total / count
    return 1 - ils

# track-to-track оценка
def evaluate_track_based(embeddings, genre_matrix, k=10):

    # матрица сходства между всеми парами элементов
    sims = cosine_similarity(embeddings)

    APs, ARs, MRRs = [], [], []
    diversity_scores = []
    all_recommended = set()

    n = len(embeddings)

    for i in range(n):

        top_k = get_top_k(sims[i], k)

        # проверка релевантности каждого трека из top-K для исходного трека
        rels = [
            is_relevant(i, j, genre_matrix)
            for j in top_k
        ]

        # m - кол-во релевантных треков для данного объекта во всем датасете 
        m = sum(
            is_relevant(i, j, genre_matrix)
            for j in range(n) if j != i
        )

        APs.append(average_precision(rels, m, k))
        ARs.append(average_recall(rels, m, k))
        MRRs.append(reciprocal_rank(rels))

        all_recommended.update(top_k)
        diversity_scores.append(diversity(embeddings[top_k]))

    return {
        "MAP@N": np.mean(APs),
        "MAR@N": np.mean(ARs),
        "MRR": np.mean(MRRs),
        "Coverage": len(all_recommended) / n,
        "Mean Diversity": np.mean(diversity_scores),
    }

# user-to-track оценка
def evaluate_user_based(user_liked_lists, embeddings, genre_matrix, k=10):

    # список векторов пользователей
    # каждый вектор - средний вектор по сохраненным трекам
    user_vectors = [
        np.mean(embeddings[likes], axis=0)
        for likes in user_liked_lists
    ]

    # матрица сходства между векторами пользователей и треков
    sims = cosine_similarity(user_vectors, embeddings)

    APs, ARs, MRRs = [], [], []
    diversity_scores = []
    all_recommended = set()

    n = len(embeddings)

    for u, likes in enumerate(user_liked_lists):

        top_k = np.argsort(sims[u])[::-1][:k]
        
        # проверка релевантности каждого трека из top-K для пользователя
        rels = [
            is_relevant_user(j, likes, genre_matrix)
            for j in top_k
        ]

        # кол-во релевантных треков для пользователя
        m = sum(
            is_relevant_user(j, likes, genre_matrix)
            for j in range(n)
        )

        APs.append(average_precision(rels, m, k))
        ARs.append(average_recall(rels, m, k))
        MRRs.append(reciprocal_rank(rels))

        all_recommended.update(top_k)
        diversity_scores.append(diversity(embeddings[top_k]))

    return {
        "MAP@N": np.mean(APs),
        "MAR@N": np.mean(ARs),
        "MRR": np.mean(MRRs),
        "Coverage": len(all_recommended) / n,
        "Mean Diversity": np.mean(diversity_scores),
    }