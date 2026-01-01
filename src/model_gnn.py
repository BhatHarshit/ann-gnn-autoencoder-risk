"""
model_gnn.py
Upgraded GNN Backbone:
 - GCN (original)
 - GraphSAGE
 - GAT
 - GATv2
Selectable via gnn_type argument.

Fully backward-compatible with your previous Hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import normalize_adj, to_torch_tensor


# ------------------------------------------------------------
# 1. Original GCN Layer
# ------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, H, A_norm):
        H_new = torch.matmul(A_norm, H)
        H_new = self.linear(H_new)
        return torch.relu(H_new)


# ------------------------------------------------------------
# 2. GraphSAGE Layer
# ------------------------------------------------------------
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, X, A_norm):
        # Mean aggregator
        neigh = torch.matmul(A_norm, X)
        cat = torch.cat([X, neigh], dim=-1)
        return torch.relu(self.linear(cat))


# ------------------------------------------------------------
# 3. GAT Layer (single head)
# ------------------------------------------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data)

    def forward(self, X, A):
        Wh = self.W(X)  # [N, out_dim]
        N = Wh.size(0)

        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, -1)

        e = torch.matmul(a_input, self.attn).squeeze(-1)
        e = torch.relu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return torch.relu(h_prime)


# ------------------------------------------------------------
# 4. GATv2 Layer (improved attention)
# ------------------------------------------------------------
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim, 1, bias=False)

    def forward(self, X, A):
        Wh = self.W(X)
        e = self.a(torch.relu(Wh))

        e = e.squeeze(-1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return torch.relu(h_prime)


# ------------------------------------------------------------
# 5. Unified Encoder
# ------------------------------------------------------------
class GNNEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=32,
                 embed_dim=16,
                 gnn_type="gcn"):    # NEW argument
        """
        gnn_type = "gcn" | "sage" | "gat" | "gatv2"
        """
        super().__init__()
        self.gnn_type = gnn_type.lower()

        # Select layers
        if self.gnn_type == "gcn":
            self.layer1 = GCNLayer(in_dim, hidden_dim)
            self.layer2 = GCNLayer(hidden_dim, embed_dim)

        elif self.gnn_type == "sage":
            self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
            self.layer2 = GraphSAGELayer(hidden_dim, embed_dim)

        elif self.gnn_type == "gat":
            self.layer1 = GATLayer(in_dim, hidden_dim)
            self.layer2 = GATLayer(hidden_dim, embed_dim)

        elif self.gnn_type == "gatv2":
            self.layer1 = GATv2Layer(in_dim, hidden_dim)
            self.layer2 = GATv2Layer(hidden_dim, embed_dim)

        else:
            raise ValueError(f"Unknown gnn_type={gnn_type}")

    def forward(self, X, A):
        # Preprocess adjacency if needed
        if isinstance(A, np.ndarray):
            A = normalize_adj(A) if self.gnn_type != "gat" else A
            A = to_torch_tensor(A, X.device)

        H = self.layer1(X, A)
        Z = self.layer2(H, A)
        return Z
