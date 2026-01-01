"""
run_experiment.py
Quick test to verify GNN encoder on SymNet data.
"""

import os
import sys
import numpy as np
import torch

# 🔧 Ensure VS Code can locate the src folder no matter where this file runs from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ✅ Import modules directly (no 'src.' prefix needed now)
from data_loader import generate_symnet
from model_gnn import GNNEncoder
from utils import normalize_adj, to_torch_tensor


def test_gnn():
    # 1. Generate synthetic financial network (SymNet)
    adj, features = generate_symnet(num_nodes=30)

    # 2. Normalize adjacency and convert to torch tensors
    A_norm = normalize_adj(adj)
    X = to_torch_tensor(features)
    A_t = to_torch_tensor(A_norm)

    # 3. Initialize GNN encoder
    model = GNNEncoder(in_dim=X.shape[1])

    # 4. Forward pass
    Z = model(X, A_t)

    # 5. Display results
    print("✅ GNN forward pass successful")
    print(f"Input feature shape  : {X.shape}")
    print(f"Output embedding shape : {Z.shape}")


if __name__ == "__main__":
    test_gnn()
