from itertools import product
from torch import nn, optim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AE(nn.Module):
    def __init__(self, n, dropout):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential (
            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, n)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
class EarlyStopping:
    def __init__(self, patience, min_delta=0.001):
        self.is_earlystopping = False
        self.best_score = float('inf')
        self.patience = patience
        self.counter=0
        self.min_delta = min_delta
        self.best_model_state = None

    def step (self, val_score, model):
        if self.best_score - val_score > self.min_delta:
            self.counter=0
            self.best_score = val_score
            self.best_model_state = copy.deepcopy(model.state_dict()) 
        else:
            self.counter+=1
            if self.counter==self.patience:
                self.is_earlystopping = True

# Визуализация функций потерь
def visualize_reconstruction(model, loader, device, num_samples=3):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        tracks = batch[0][:num_samples].to(device)

        reconstructed = model(tracks)
        orig = tracks.cpu().numpy()
        recon = reconstructed.cpu().numpy()

        fig, axs = plt.subplots(1, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            axs[i].plot(orig[i], label='Original', alpha=0.7)
            axs[i].plot(recon[i], label='Reconstructed', alpha=0.7)
            axs[i].set_title(f"Track {i+1}")
            axs[i].set_xlabel("Feature")
            axs[i].set_ylabel("Value")
            axs[i].legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
# построение сравнительных примеров исходных данных и реконструированных
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

# Обучение одной конфигурации модели
def train_one_fold(model, optimizer, train_loader, val_loader, loss_function, epochs=5, device='cuda'):
    import matplotlib.pyplot as plt
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor = 0.5, 
        patience = 5 
    )
    train_losses_arr = []
    val_losses_arr = []
    earlystopping = EarlyStopping(patience=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range (epochs):
        print ("epoch:", epoch+1)
        train_losses = 0.0
        model.train()
        for batch in train_loader:
            tracks = batch[0].to(device)
            # Строится матрица [32, 518]
            train_reconstructed = model(tracks)
            # [32, 518]
            train_loss = loss_function(train_reconstructed, tracks)

            optimizer.zero_grad()
            train_loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step() 

            train_losses += torch.mean((tracks - train_reconstructed) ** 2).item()
        avg_train_losses = train_losses/len(train_loader)
        train_losses_arr.append(avg_train_losses)

        val_losses = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                tracks = batch[0].to(device)
                val_reconstructed = model(tracks)
                val_loss = loss_function(val_reconstructed, tracks)
                val_losses += torch.mean((tracks - val_reconstructed) ** 2).item()

            avg_val_losses = val_losses/len(val_loader)
            val_losses_arr.append(avg_val_losses)
            scheduler.step(avg_val_losses)

        earlystopping.step(avg_val_losses, model)
        if earlystopping.is_earlystopping == True:
            model.load_state_dict(earlystopping.best_model_state)
            print (f"Early Stopping on {epoch + 1} epoch")
            break

    train_val_visual (train_losses_arr, val_losses_arr)
    
    return earlystopping.best_score

def get_latent_vectors (model, loader, device):
        model.eval()
        
        latent_vectors = []
        with torch.no_grad():
            for batch in loader:
                tracks = batch[0].to(device)
                encoded_tracks = model.encoder(tracks)
                latent_vectors.append(encoded_tracks.cpu().numpy())
        return np.vstack(latent_vectors)

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

# Полный перебор всех комбинаций гиперпараметров
def full_process (train_df, val_df, path, features): # Тензор - специальный формат, чтоб нейросети могли его обрабатывать
    train_data_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    # Чтобы тензор можно было скормить в загрузчик
    train_dataset_tensor = TensorDataset(train_data_tensor)
    train_loader = DataLoader(dataset=train_dataset_tensor, batch_size=32, shuffle=True)
    
    val_data_tensor = torch.tensor(val_df.values, dtype=torch.float32)
    val_dataset_tensor = TensorDataset(val_data_tensor)
    val_loader = DataLoader(dataset=val_dataset_tensor, batch_size=32, shuffle=False)
    
    param_grid = {
        'lr': [1e-5, 1e-3],
        'dropout': [0.2, 0.3],
        'loss_name': ['MSE', 'SmoothL1']
    }
    
    best_model = None
    best_val_loss = float('inf')
    best_parameters = None
    
    for lr, dropout, loss_name in product(*param_grid.values()):
        model = AE(features, dropout)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print (f"Current parameters: learning rate = {lr}, dropout = {dropout}, loss function = {loss_name}")
        if loss_name == 'MSE':
            loss_function = nn.MSELoss()
        else:
            loss_function = nn.SmoothL1Loss(beta=0.5)
    
        val_loss = train_one_fold(model, optimizer, train_loader, val_loader, loss_function, epochs=50, device=device)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_parameters = {'lr': lr, 'dropout': dropout, 'loss_name': loss_name}
    
    best_model = AE(features, dropout=best_parameters['dropout'])
    best_model.load_state_dict(best_model_state)
            
    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'dropout': best_parameters['dropout'],
        'lr': best_parameters['lr'],
        'loss_name': best_parameters['loss_name'],
        'val_loss': best_val_loss
    }
    
    torch.save(checkpoint, path)
    
    checkpoint = torch.load(path, map_location=device)
    best_model = AE(features, checkpoint['dropout'])
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    best_model.eval()

    all_tracks = pd.concat([X_train, X_val, X_test])
    data_tensor = torch.tensor(all_tracks.iloc[:, :features].values, dtype=torch.float32)
    dataset_tensor = TensorDataset(data_tensor)
    loader = DataLoader(dataset=dataset_tensor, batch_size=32, shuffle=False)

    test_data_tensor = torch.tensor(test_df.iloc[:, :features].values, dtype=torch.float32)
    test_dataset_tensor = TensorDataset(test_data_tensor)
    test_loader = DataLoader(dataset=test_dataset_tensor, batch_size=32, shuffle=False)
    
    embeddings_np = get_latent_vectors (best_model, loader, device)
    test_embeddings_np = get_latent_vectors (best_model, test_loader, device)
    if num_features == 518:
        np.save('ae_audio_embeddings.npy', embeddings_np)
        np.save('ae_audio_test_embeddings.npy', test_embeddings_np)
    else:
        np.save('ae_hybrid_embeddings.npy', embeddings_np)
        np.save('ae_hybrid_test_embeddings.npy', test_embeddings_np)
        
    visualize_reconstruction(best_model, val_loader, device)
    print(f"Best parameters: {best_parameters}")
    print(f"Best val loss: {best_val_loss:.4f}")

import copy
print ("Without genre columns")
full_process(train_df[audio_cols], val_df[audio_cols], "best_au_full.pth", 518)

import copy
print ("With genre columns")
full_process(train_df, val_df, "best_au_full_with_genres.pth", 535)
