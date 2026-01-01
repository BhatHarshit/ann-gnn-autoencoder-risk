"""
train_hybrid.py
Training loop for Hybrid GNN + Autoencoder model on SymNet dataset.
"""

import os
import sys
import torch
import torch.optim as optim

# 🔧 Ensure the current directory is added to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from data_loader import generate_symnet
from utils import normalize_adj, to_torch_tensor
from model_hybrid import HybridGNN_Autoencoder


def train_hybrid_model(epochs=50, lr=0.001, save_path="experiments/hybrid_model.pt"):
    # 1. Generate synthetic dataset
    adj, features = generate_symnet(num_nodes=30)
    A_norm = normalize_adj(adj)

    X = to_torch_tensor(features)
    A_t = to_torch_tensor(A_norm)

    # 2. Initialize model and optimizer
    model = HybridGNN_Autoencoder(in_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        X_recon, Z = model(X, A_t)
        loss = model.reconstruction_loss(X_recon, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    # 4. Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved at {save_path}")


if __name__ == "__main__":
    train_hybrid_model()
