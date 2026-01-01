"""
utils.py
Utility functions for graph normalization and tensor conversion.
"""

import numpy as np
import torch

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2)
    """
    A = adj + np.eye(adj.shape[0])  # add self-loops
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def to_torch_tensor(np_array, device=None):
    """
    Convert a NumPy array to a PyTorch tensor.
    """
    tensor = torch.tensor(np_array, dtype=torch.float32)
    if device:
        tensor = tensor.to(device)
    return tensor
