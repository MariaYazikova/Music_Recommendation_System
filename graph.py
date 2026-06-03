from itertools import product
from torch import nn, optim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import copy
from sklearn.neighbors import NearestNeighbors
import gc
import faiss
import random
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.nn import Dropout
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv 

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNN (nn.Module):
    def __init__(self, num_features, dropout):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 128, aggr='mean')
        self.conv2 = GCNConv(128, 64, aggr='mean')
        self.bn1 = nn.BatchNorm1d(128) # BatchNorm тоже можно добавить!
        self.dropout = Dropout(dropout)
    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            
        x = self.bn1(x)
        x = F.relu(x)
        x = F.normalize(x, p=2, dim=1) 
        x = self.dropout(x)

        if edge_weight is not None:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        
        return x     

class EarlyStopping:
    def __init__(self, plateau_threshold=0.0001, plateau_window=5):
        self.is_earlystopping = False
        self.best_score = float('inf')
        self.counter = 0
        self.best_model_state = None

        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.loss_history = []

    def step (self, val_score, model):
        self.loss_history.append(val_score)
        self.counter+=1
        if self.counter >= self.plateau_window:
            recent_std = float(np.std(self.loss_history[-self.plateau_window:]))
            if recent_std < self.plateau_threshold:
                self.best_score = val_score
                self.is_earlystopping = True
                self.best_model_state = copy.deepcopy(model.state_dict())
def replace_id_with_pos (arr, is_index=False):
    if is_index:
        arr = arr.index.to_list()
    return {index: pos for pos, index in enumerate(arr)}

def replace_pos_with_id (id_to_position):
    return {pos: index for index, pos in id_to_position.items()}

def make_graph (df, num_features, k_neighbours=10): # Для каждого жанра определяем все треки, которые ему принадлежат
    node_features = df.iloc[:, :num_features].values

    x = torch.tensor(node_features, dtype=torch.float32)
    n_nodes = len(df)

    # Нормализуем данные для косинусного сходства 
    node_features_norm = node_features / np.linalg.norm(node_features, axis=1, keepdims=True)
    node_features_norm = node_features_norm.astype(np.float32) # Faiss любит float32
    
    # Создаем индекс Faiss
    d = node_features_norm.shape[1] 
    index = faiss.IndexFlatIP(d)  
    index.add(node_features_norm)   
    
    # Ищем соседей
    D, I = index.search(node_features_norm, k_neighbours + 1)
    
    # D - расстояния (сходства), I - индексы соседей
    distances = 1 - D 
    indices = I    
        
    similarities = 1 - distances
    raw_edges = []
    for i in tqdm(range(n_nodes)):
        for k_idx in range(1, k_neighbours + 1):
            j = indices[i][k_idx]
            score = similarities[i][k_idx]
            
            if score > 0: 
                raw_edges.append((i, j, float(score)))

        if i % 10000 == 0:
            gc.collect()
            
    edges_by_type = {'train': [], 'val': [], 'test': []}
    weights_by_type = {'train': [], 'val': [], 'test': []}

    edge_weights = defaultdict(lambda: {'count': 0, 'splits': set()})
    seen_pairs = set()
    for u, v, weight in raw_edges:
        split_u = pos_to_split[u]
        split_v = pos_to_split[v]

        if split_u == 'train' and split_v == 'train':
            edge_type = 'train'
        elif split_u == 'val' or split_v == 'val':
            edge_type = 'val'
        else:
            edge_type = 'test'

        pair_key = tuple(sorted((u, v)))
        
        if pair_key not in seen_pairs:
            # Добавляем двунаправленные ребра
            edges_by_type[edge_type].append([u, v])
            weights_by_type[edge_type].append(weight)
            
            edges_by_type[edge_type].append([v, u])
            weights_by_type[edge_type].append(weight)
            
            seen_pairs.add(pair_key)
    del node_features, distances, indices, similarities
    gc.collect()

    return x, edges_by_type, weights_by_type
        
def list_to_tensor (edge_list, weight_list):
    e = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    w = torch.tensor(weight_list, dtype=torch.float32, device=device)
    return e, w


def make_data (df, num_features):
    x, edges, weights = make_graph (df, num_features)

    train_edge_index, train_edge_weight = list_to_tensor(edges['train'], weights['train'])
    val_edge_index, val_edge_weight = list_to_tensor(edges['val'], weights['val'])
    test_edge_index, test_edge_weight = list_to_tensor(edges['test'], weights['test'])
    full_edge_index, full_edge_weight = list_to_tensor (edges['train'] + edges['val'] + edges['test'], weights['train'] + weights['val'] + weights['test'])
    
    # Создаем единый объект, который понимает PyTorch Geometric
    data = Data(
        x=x.to(device), 
        edge_index = train_edge_index, 
        edge_attr = train_edge_weight
    )
    full_data = Data(
        x=x.to(device),
        edge_index = full_edge_index, 
        edge_attr = full_edge_weight 
    )
    return data, val_edge_index, test_edge_index, full_data

def train_val_visual (train_losses_arr, val_losses_arr):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs = axs.ravel()
    titles = ['Train losses', 'Val losses']
    losses = [train_losses_arr, val_losses_arr]
    for ax, title, loss in zip(axs, titles, losses):
        ax.plot(loss, label=title)
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()

def train_one_fold (df, train_data, val_edge_index, num_features, model, optimizer, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device) # объект data из предыдущих шагов
    
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor = 0.5,
        patience = 5 

    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    epochs = 50
    loss_history_train = []
    loss_history_val = []
    earlystopping = EarlyStopping()
    
    for epoch in range(epochs):
        train_losses = 0.0
        model.train()
        print ("epoch", epoch+1)
        num_batches = 0
        optimizer.zero_grad()

        embeddings = model(train_data.x, train_data.edge_index, train_data.edge_attr)
    
        src = embeddings[train_data.edge_index[0]]
        dst = embeddings[train_data.edge_index[1]]

        pos_loss = F.mse_loss(src, dst)
        
        num_neg_samples = train_data.edge_index.size(1)
        neg_edge_index = negative_sampling( 
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=num_neg_samples
        )
        
        neg_src = embeddings[neg_edge_index[0]]
        neg_dst = embeddings[neg_edge_index[1]]

        neg_target = torch.ones(neg_src.size(0), device=device) * -1
        neg_loss = F.cosine_embedding_loss(neg_src, neg_dst, neg_target)
        
        loss = pos_loss + neg_loss
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
            
        avg_train_loss = loss.item()
        loss_history_train.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            full_embeddings = model(train_data.x, train_data.edge_index, train_data.edge_weight)

            val_src = full_embeddings[val_edge_index[0]]
            val_dst = full_embeddings[val_edge_index[1]]

            val_loss = F.mse_loss(val_src, val_dst).item()
            
        loss_history_val.append(val_loss)
        scheduler.step(val_loss)
        earlystopping.step(val_loss, model)
        if earlystopping.is_earlystopping == True:
            model.load_state_dict(earlystopping.best_model_state)
            print (f"Early Stopping on {epoch + 1} epoch")
            break

    train_val_visual(loss_history_train, loss_history_val)
    return earlystopping.best_score
        
        
def full_process (df, num_features, path, test_path):
    train_data, val_edge_index, test_edge_index, full_data = make_data (df, num_features)

    param_grid = {
        'lr': [1e-4, 1e-3],
        'dropout': [0.2, 0.5]
    }
    
    best_model = None
    best_val_loss = float('inf')
    best_parameters = None
    
    for lr, dropout, in product(*param_grid.values()):
        model = GNN(num_features, dropout)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print (f"Current parameters: learning rate = {lr}, dropout = {dropout}")
    
        val_loss = train_one_fold(df, train_data, val_edge_index, num_features, model, optimizer, device=device)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_parameters = {'lr': lr, 'dropout': dropout}

    # Финальное получение эмбеддингов
    final_model = GNN(num_features, best_parameters['dropout'])
    final_model.load_state_dict(best_model_state)
    final_model.to(device)
    final_model.eval()
    
    final_data = full_data.to(device)
    
    with torch.no_grad():
        embeddings = final_model(full_data.x, full_data.edge_index, full_data.edge_attr)
    
    np.save(path, embeddings.cpu().numpy())
        
    test_embeddings = embeddings[test_pos]
    np.save(test_path, test_embeddings.cpu().numpy())

    print(f"Best parameters: {best_parameters}")
    print(f"Best val loss: {best_val_loss:.4f}")

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'
all_tracks = pd.concat([train_df, val_df, test_df], ignore_index = False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

id_to_pos = replace_id_with_pos(all_tracks, True)
with open('id_to_pos.pkl', 'wb') as a:
    pickle.dump(id_to_pos, a)
pos_to_id = replace_pos_with_id (id_to_pos)

with open('pos_to_id.pkl', 'wb') as b:
    pickle.dump(pos_to_id, b)

pos_to_split = {pos: row['split'] for pos, (track_id, row) in enumerate(all_tracks.iterrows())}
test_pos = [pos for pos, split in pos_to_split.items() if split == 'test']

test_ids = [pos_to_id[pos] for pos in test_pos]
with open('test_track_ids.pkl', 'wb') as c:
        pickle.dump(test_ids, c)

full_process (all_tracks, 518, 'all_embeddings.npy', 'test_embeddings.npy')
full_process (all_tracks, 535, 'all_embeddings_hybrid.npy', 'test_embeddings_hybrid.npy')
