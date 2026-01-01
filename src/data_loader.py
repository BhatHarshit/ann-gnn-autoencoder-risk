"""
data_loader.py
Generates or loads financial network data (SymNet synthetic dataset)

Provides:
- generate_symnet(...)
- generate_symnet_large(...)
- inject_synthetic_anomalies(...)
- CLI interface
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse


# ============================================================
# SMALL ORIGINAL SYMNET (DEFAULT)
# ============================================================
def generate_symnet(num_nodes=30, connection_prob=0.2, random_seed=42, feature_dim=6, save=True):
    """
    Original small SymNet generator.
    DEFAULT FIXED: feature_dim changed from 3 → 6 for compatibility with models.
    Returns: adj_matrix, features
    """
    np.random.seed(random_seed)

    G = nx.erdos_renyi_graph(num_nodes, connection_prob, seed=random_seed)
    adj_matrix = nx.to_numpy_array(G, dtype=float)

    features = np.random.rand(num_nodes, feature_dim).astype(np.float32)
    cols = [f"feat_{i}" for i in range(feature_dim)]
    feature_df = pd.DataFrame(features, columns=cols)

    if save:
        os.makedirs("data", exist_ok=True)
        np.savez(f"data/symnet_data_{num_nodes}.npz",
                 adjacency=adj_matrix,
                 features=features)
        feature_df.to_csv(f"data/symnet_data_{num_nodes}.csv", index=False)

    print(f"✅ Generated SymNet dataset with {num_nodes} nodes")
    print(f"Average degree: {np.mean([deg for _, deg in G.degree()]):.2f}")
    print(f"Adjacency shape: {adj_matrix.shape}")

    return adj_matrix, features


# ============================================================
# LARGE NETWORK GENERATOR
# ============================================================
def generate_symnet_large(num_nodes=1000, avg_degree=6, feature_dim=6, random_seed=42, save=True):
    """
    Generate a large synthetic financial network.
    """
    np.random.seed(random_seed)

    p = float(avg_degree) / max(1, (num_nodes - 1))
    G = nx.erdos_renyi_graph(num_nodes, p, seed=random_seed)
    adj = nx.to_numpy_array(G, dtype=float)

    # Features: N x feature_dim normal distribution
    features = np.random.normal(0, 1, (num_nodes, feature_dim)).astype(np.float32)
    features += np.random.normal(0, 0.1, size=(num_nodes, 1))  # bias shift

    cols = [f"feat_{i}" for i in range(feature_dim)]
    feature_df = pd.DataFrame(features, columns=cols)

    if save:
        os.makedirs("data", exist_ok=True)
        np.savez(f"data/symnet_data_{num_nodes}.npz",
                 adjacency=adj,
                 features=features)
        feature_df.to_csv(f"data/symnet_data_{num_nodes}.csv", index=False)

    print(f"✅ Generated LARGE SymNet dataset with {num_nodes} nodes")
    print(f"Target avg degree: {avg_degree} (p={p:.6f})")
    print(f"Actual avg degree: {np.mean([d for _, d in G.degree()]):.2f}")
    print(f"Adjacency shape: {adj.shape}")

    return adj, features


# ============================================================
# ANOMALY INJECTION
# ============================================================
def inject_synthetic_anomalies(adj, features, anomaly_fraction=0.02,
                               magnitude=5.0, seed=1, save=True):
    """
    Inflate features for randomly selected nodes to create anomalies.
    Returns: new_features, labels, anomaly_indices
    """
    np.random.seed(seed)

    N = features.shape[0]
    num_anom = max(1, int(N * anomaly_fraction))
    anom_idx = np.random.choice(N, num_anom, replace=False)

    features_new = features.copy()

    noise = np.abs(np.random.normal(1.0, 0.5, (num_anom, features.shape[1])))
    features_new[anom_idx] += magnitude * noise

    labels = np.zeros(N, dtype=int)
    labels[anom_idx] = 1

    if save:
        os.makedirs("data", exist_ok=True)
        base = f"data/symnet_data_{N}"

        np.savetxt(base + "_labels.csv", labels, fmt="%d")
        np.savetxt(base + "_anomaly_idx.csv", anom_idx, fmt="%d")

        pd.DataFrame(features_new,
                     columns=[f"feat_{i}" for i in range(features.shape[1])]).to_csv(
            base + "_features_injected.csv", index=False
        )

    print(f"✅ Injected {num_anom} anomalies (fraction={anomaly_fraction})")
    print(f"Anomaly index sample: {anom_idx[:10].tolist()}")

    return features_new, labels, anom_idx


# ============================================================
# CLI
# ============================================================
def _cli():
    parser = argparse.ArgumentParser(description="SymNet generator with optional anomaly injection.")

    parser.add_argument("--large", type=int, default=0,
                        help="Generate LARGE SymNet with N nodes.")
    parser.add_argument("--avg_degree", type=float, default=6.0,
                        help="Average degree target (only for large).")
    parser.add_argument("--feature_dim", type=int, default=6,
                        help="Feature dimension for nodes.")
    parser.add_argument("--inject", type=float, default=0.0,
                        help="Inject anomaly fraction, e.g., 0.02.")
    parser.add_argument("--magnitude", type=float, default=5.0,
                        help="Anomaly magnitude multiplier.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for anomaly injection.")

    args = parser.parse_args()

    # --- Large graph ---
    if args.large > 0:
        adj, feat = generate_symnet_large(
            num_nodes=args.large,
            avg_degree=args.avg_degree,
            feature_dim=args.feature_dim
        )
    # --- Small graph ---
    else:
        adj, feat = generate_symnet(
            feature_dim=args.feature_dim
        )

    # Inject anomalies
    if args.inject > 0:
        feat, labels, idx = inject_synthetic_anomalies(
            adj, feat,
            anomaly_fraction=args.inject,
            magnitude=args.magnitude,
            seed=args.seed
        )


if __name__ == "__main__":
    _cli()
