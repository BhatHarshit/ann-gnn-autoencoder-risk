# src/model_gnn_autoencoder.py

import torch
import torch.nn as nn
from model_gnn import GNNEncoder

class GNNAutoEncoder(nn.Module):
    def __init__(self, in_dim=6, gnn_hidden=64, embed_dim=32):
        super().__init__()
        self.gnn = GNNEncoder(in_dim=in_dim,
                              hidden_dim=gnn_hidden,
                              embed_dim=embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim)
        )

    def forward(self, X, A):
        Z = self.gnn(X, A)
        X_hat = self.decoder(Z)
        return X_hat, Z
