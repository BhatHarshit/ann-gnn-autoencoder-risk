"""
visualize_embeddings.py
Projects learned node embeddings (Z) into 2D space using PCA for visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_loader import generate_symnet
from utils import normalize_adj, to_torch_tensor
from model_hybrid import HybridGNN_Autoencoder

# Load trained model
model = HybridGNN_Autoencoder(in_dim=3)
model.load_state_dict(torch.load("experiments/hybrid_model.pt"))
model.eval()

# Load data
adj, features = generate_symnet(num_nodes=30)
A_norm = normalize_adj(adj)
X = to_torch_tensor(features)
A_t = to_torch_tensor(A_norm)

# Forward pass
with torch.no_grad():
    X_recon, Z = model(X, A_t)
    Z_np = Z.detach().numpy()

# PCA for 2D projection
pca = PCA(n_components=2)
Z_2d = pca.fit_transform(Z_np)

plt.figure(figsize=(6,5))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c='blue', s=60)
plt.title("Node Embedding Visualization (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("experiments/results/node_embeddings.png")
plt.show()
print("📈 Saved node embeddings: experiments/results/node_embeddings.png")
