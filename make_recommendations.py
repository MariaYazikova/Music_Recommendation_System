def load_files():
    import pickle
    import numpy as np
    cfg_path = "data_for_recommendations/best_knn_hybrid_config.pkl"
    with open(cfg_path, "rb") as f:
        config = pickle.load(f)
    
    ae_embeddings_np = np.load('data_for_recommendations/ae_hybrid_embeddings.npy')
    
    id_to_pos_path = 'data_for_recommendations/id_to_pos.pkl'
    with open(id_to_pos_path, "rb") as f:
        id_to_pos = pickle.load(f)
    
    pos_to_id_path = 'data_for_recommendations/pos_to_id.pkl'
    with open(pos_to_id_path, "rb") as f:
        pos_to_id = pickle.load(f)

def user_recommendation (likes):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(
        n_neighbors=50,
        metric=config.get("metric", "cosine"),
        algorithm=config.get("algorithm", "auto")
    )
    
    knn.fit(ae_embeddings_np)
    
    favorite_poses = [id_to_pos[track_id] for track_id in likes]
    favorite_vectors = ae_embeddings_np[favorite_poses]
    user_emb = np.mean(favorite_vectors, axis=0)
    user_emb = user_emb.reshape(1, -1)
            
    dist, idx = knn.kneighbors(user_emb, n_neighbors=50)
    rec_poses = idx[0]
    rec_indices = [pos_to_id[track_pos] for track_pos in rec_poses]
    return rec_indices

def track_recommendation (track):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(
        n_neighbors=50,
        metric=config.get("metric", "cosine"),
        algorithm=config.get("algorithm", "auto")
    )
    
    knn.fit(siam_embeddings_np)
    track_pos = id_to_pos[track]
    track_emb = siam_embeddings_np[track_pos]
    track_emb = track_emb.reshape(1, -1)
            
    dist, idx = knn.kneighbors(track_emb, n_neighbors=50)
    rec_poses = idx[0]
    rec_indices = [pos_to_id[track_pos] for track_pos in rec_poses]
    return rec_indices
