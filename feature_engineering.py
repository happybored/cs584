import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler
import torch.optim as optim
import numpy as np
import sys
from lut import Sorted_LUT
from so6 import SO6,SO6_basis
from scipy.linalg import logm
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SO6Dataset(Dataset):
    def __init__(self, X, y):
        """
        X: (num_samples, seq_len=36, input_dim=3)
        y: (num_samples, 1)  # Target values (Regression)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class MLP(nn.Module):
    def __init__(self, in_dim=36*3, out_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc4 = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc3 = nn.Linear(512, out_dim )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



def split_train_val(X, y, train_ratio=0.5, seed=47):
    """
    Randomly shuffle and split dataset into training and validation sets.
    
    Parameters:
        X (np.array): Input features of shape (num_samples, seq_len, input_dim)
        y (np.array): Target values of shape (num_samples, 1)
        train_ratio (float): Percentage of data to use for training (default 80%)
        seed (int): Random seed for reproducibility
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    np.random.seed(seed)  # Set seed for reproducibility
    indices = np.random.permutation(len(X))  # Shuffle indices
    split_idx = int(train_ratio * len(X))  # Compute split index

    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]


    return X_train, y_train, X_val, y_val


def train_model_classification(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, device="cpu"):
    """
    Train Transformer model on SO(6) data using GPU.
    """
    model.to(device)  # Move model to GPU

    criterion = nn.CrossEntropyLoss() #nn.MSELoss()  # Mean Squared Error Loss (for regression)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-6 )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()  # 🔹 Convert labels to `long`

            optimizer.zero_grad()
            y_pred = model(X_batch)  # Shape: (batch_size, num_classes)
            y_batch = y_batch.float()
            loss =  F.binary_cross_entropy_with_logits(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        scheduler.step()

        # avg_train_loss = total_loss 

        # -------------------------------
        # Validation Step
        # -------------------------------
        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
                
                y_pred = model(X_batch)  # Shape: (batch_size, num_classes)
                y_pred_class = torch.argmax(y_pred, dim=1)  # 🔹 Get predicted class
                for i in range(y_pred_class.shape[0]):
                    if y_batch[i][y_pred_class[i]] == 1:
                        total_correct += 1
                total_samples += y_batch.size(0)

        accuracy = total_correct / total_samples * 100  # Compute accuracy

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    

def binary_encoding(x, n_bits=4):
    """
    x: (N, k) array of signed integers
    return: (N, k * n_bits) binary encoding using two's complement
    """
    x = x.astype(np.int32)
    N, k = x.shape

    # Two's complement trick: mask with unsigned equivalent
    unsigned = x & (2**n_bits - 1)

    bits = ((unsigned[:, :, None] >> np.arange(n_bits)) & 1).astype(np.float32)
    return bits.reshape(N, k * n_bits)



def fourier_features(x, n_bits=4):
    """
    x: (N, k) array of integers
    return: (N, 2 * k) array with [sin(2πx/2ⁿ), cos(2πx/2ⁿ)] per element
    """
    x_scaled = 2 * np.pi * x / (2 ** n_bits)  # shape: (N, k)
    return np.concatenate([np.sin(x_scaled), np.cos(x_scaled)], axis=1).astype(np.float32)


def plot_data(_data,_ylabel):
    plt.figure(1)
    _data = torch.tensor(_data, dtype=torch.float)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel(_ylabel)
    plt.plot(_data.numpy())
    plt.show()


# -------------------------------
# Main Script (Auto GPU Detection)
# -------------------------------
def main():
    with open('output/X_after_perm.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('output/y_feasable_onehot.pkl', 'rb') as f:
        y_feasable = pickle.load(f)

    with open('output/X_lde_pattern.pkl', 'rb') as f:
        X_lde = pickle.load(f)

    X = np.array(X)  
    X_parity = X[:,:,1]
    X_parity = X_parity%2
    X = X.reshape(X.shape[0], 36 * 3)
    Xb = binary_encoding(X,n_bits=4)
    


    ## just change the following step:
    X = np.concatenate([X], axis=1)  #original
    #X = np.concatenate([X_parity], axis=1)  #parity
    #X = np.concatenate([X_lde], axis=1)   #lde
    X = np.concatenate([Xb], axis=1)  #binary encoding
    #X = np.concatenate([X_parity，X_lde], axis=1)  #combinations
    #X = np.concatenate([X,X_parity，X_lde], axis=1) 

    X = np.pad(X, ((0, 0), (0, 600 - X.shape[1])), mode='constant', constant_values=0)

    y = y_feasable
    print(X.shape)
    print(y.shape)

    # split train/val
    X_train, y_train, X_val, y_val = split_train_val(X,y)


    # Create dataset
    train_dataset = SO6Dataset(X_train, y_train)
    test_dataset =  SO6Dataset(X_val, y_val)


    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    print(N_train, N_val)
    
    
    # Create DataLoaders for Train & Validation
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(test_dataset, batch_size=256)


    # Initialize Model
    model = MLP(in_dim=X_train.shape[1],out_dim=15)

    # Train Model on GPU
    train_model_classification(model, train_loader, val_loader, num_epochs=300, learning_rate=1e-3, device=device)

    torch.save(model.state_dict(), 'model/step_net_afterperm.pth')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main()