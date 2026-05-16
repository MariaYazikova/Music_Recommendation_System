import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

    for k, r in enumerate(rels, 1):

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

    for k, r in enumerate(rels, 1):

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
def evaluate_track_based(embeddings_np, genre_matrix_np, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = torch.from_numpy(embeddings_np).float().to(device)      # (N, 32)
    genres = torch.from_numpy(genre_matrix_np).float().to(device)        # (N, M)

    APs, ARs, MRRs = [], [], []
    diversity_scores = []
    all_recommended = set()

    n_tracks = embeddings.shape[0]

    for i in tqdm.tqdm(range(n_tracks)):
        # 1. Берем вектор текущего трека
        track_vec = embeddings[i].unsqueeze(0) # (1, 32)
        
        # 2. Считаем сходство со всеми треками сразу (Dot Product == Cosine Sim для нормализованных векторов)
        sims = torch.mm(track_vec, embeddings.T)[0] # (N,)

        # Маскируем лайкнутые треки (ставим им очень низкий скор, чтобы они не попали в топ)
        mask = torch.ones(n_tracks, device=device)
        mask[i] = -1e9 
        masked_sims = sims + mask

        # 4. Берем топ-K рекомендаций
        rec_vals, rec_poses = torch.topk(masked_sims, k=k) # (K,)
        recommended_positions = rec_poses.cpu().numpy()

        # A. Релевантность рекомендаций (rels)
        rec_genres = genres[rec_poses]             # (K, M)
        liked_genres = genres[i].unsqueeze(0)       # (1, M) - жанры исходного трека
        
        # Проверяем пересечение жанров у рекомендованных треков с маской пользователя
        # (K, M) * (1, M) -> (K, M). any(dim=1) -> (K,) boolean
        relevances_bool = (rec_genres * liked_genres.unsqueeze(0)).any(dim=1)
        rels = relevances_bool.int().cpu().numpy()[0] # (K,) array of 0/1
        
        # B. Расчет m (сколько всего треков в базе релевантны пользователю?)
        # Вместо цикла for j in range(n_tracks):
        # Мы умножаем матрицу всех жанров (N, M) на вектор маски (M, 1)
        # Результат (N, 1): для каждого трека сумма общих жанров с пользователем
        overlaps_all = torch.mm(genres, liked_genres.T)
        
        # Трек релевантен, если overlap > 0
        is_relevant_global = (overlaps_all > 0).squeeze(1) # (N,) boolean
        
        # m = количество таких треков
        m = is_relevant_global.sum().item()

        APs.append(average_precision(rels, m, k))
        ARs.append(average_recall(rels, m, k))
        MRRs.append(reciprocal_rank(rels))

        # ИСПРАВЛЕНИЕ: Превращаем позиции обратно в ID треков
        recommended_ids = [pos_to_id[pos] for pos in recommended_positions]
        
        all_recommended.update(recommended_ids)
        rec_embs_cpu = embeddings[recommended_positions].cpu().numpy()
        diversity_scores.append(diversity(rec_embs_cpu))

    return {
        "MAP@N": np.mean(APs),
        "MAR@N": np.mean(ARs),
        "MRR": np.mean(MRRs),
        "Coverage": len(all_recommended) / n_tracks,
        "Mean Diversity": np.mean(diversity_scores),
    }

# user-to-track оценка
# В user_liked_lists добавлен столбец 'vector' с уже обработанным эмбеддингом пользователя, это тензор размера (1, n), где n - количество признаков в эмбэддинге
def evaluate_user_based(user_liked_lists, embeddings_np, genre_matrix_np, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    APs, ARs, MRRs = [], [], []
    diversity_scores = []
    all_recommended = set()

    # 1. Переносим все данные на GPU один раз
    embeddings = torch.from_numpy(embeddings_np).float().to(device)      # (N, 32)
    genres = torch.from_numpy(genre_matrix_np).float().to(device)        # (N, M)

    n_tracks = embeddings.shape[0]

    for user in tqdm.tqdm(user_liked_lists):
        user_emb = user['vector']
        # --- ШАГ 4: Поиск рекомендаций (User Embedding vs All Tracks) ---
        sims = torch.mm(user_emb, embeddings.T)[0] # (N,)

        favorite_poses = [id_to_pos[track_id] for track_id in user['liked_tracks']]
        # Превращаем список позиций в тензор позиций
        liked_poses_tensor = torch.tensor(favorite_poses, dtype=torch.long, device=device)

        # Маскируем лайкнутые треки (ставим им очень низкий скор, чтобы они не попали в топ)
        mask = torch.ones(n_tracks, device=device)
        mask[liked_poses_tensor] = -1e9 
        masked_sims = sims + mask

        # Берем топ-K рекомендаций
        rec_vals, rec_poses = torch.topk(masked_sims, k=k) # (K,)
        recommended_positions = rec_poses.cpu().numpy()    # Переносим на CPU для дальнейшей работы с pandas/sets

        # A. Релевантность рекомендаций (rels)
        rec_genres = genres[rec_poses]             # (K, M)
        liked_genres = genres[liked_poses_tensor]  # (L, M)
        
        # Маска жанров пользователя: какие жанры есть хотя бы в одном лайке
        user_genre_mask = liked_genres.any(dim=0)    # (M,) boolean
        
        # Проверяем пересечение жанров у рекомендованных треков с маской пользователя
        # (K, M) * (1, M) -> (K, M). any(dim=1) -> (K,) boolean
        relevances_bool = (rec_genres * user_genre_mask.unsqueeze(0)).any(dim=1)
        rels = relevances_bool.int().cpu().numpy() # (K,) array of 0/1
        
        # B. Расчет m (сколько всего треков в базе релевантны пользователю?)
        # Вместо цикла for j in range(n_tracks):
        # Мы умножаем матрицу всех жанров (N, M) на вектор маски (M, 1)
        # Результат (N, 1): для каждого трека сумма общих жанров с пользователем
        overlaps_all = torch.mm(genres, user_genre_mask.float().unsqueeze(1))
        
        # Трек релевантен, если overlap > 0
        is_relevant_global = (overlaps_all > 0).squeeze(1) # (N,) boolean
        
        # m = количество таких треков
        m = is_relevant_global.sum().item()

        APs.append(average_precision(rels, m, k))
        ARs.append(average_recall(rels, m, k))
        MRRs.append(reciprocal_rank(rels))

        # ИСПРАВЛЕНИЕ: Превращаем позиции обратно в ID треков
        recommended_ids = [pos_to_id[pos] for pos in recommended_positions]
        
        # Добавляем эти ID в общее множествоA
        all_recommended.update(recommended_ids)
        rec_embs_cpu = embeddings[recommended_positions].cpu().numpy()
        diversity_scores.append(diversity(rec_embs_cpu))

    return {
        "MAP@N": np.mean(APs),
        "MAR@N": np.mean(ARs),
        "MRR": np.mean(MRRs),
        "Coverage": len(all_recommended) / n_tracks,
        "Mean Diversity": np.mean(diversity_scores),
    }
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
