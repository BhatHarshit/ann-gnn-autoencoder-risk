# src/dataset_builder.py
"""
Dataset builder for faithful SysNet-10, SysNet-40, EURNET proxy, and doubled/augmented datasets.
Usage (from repo root, with virtualenv active):
    python src/dataset_builder.py
    python src/dataset_builder.py --num_nodes 1000 --seed 42
    python src/dataset_builder.py --eurnet data/eurnet.csv    # if you have an EURNET CSV
Outputs (data/):
    data/symnet10_original.npz, .csv
    data/symnet40_original.npz, .csv
    data/eurnet_original.npz, .csv  (or proxy)
    data/symnet10_augmented_{variant}.npz / csv
    data/combined/combined_dataset.npz
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx

np.random.seed(42)

DATA_DIR = "data"
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)


def build_graph(num_nodes, target_avg_degree, seed):
    p = float(target_avg_degree) / max(1, (num_nodes - 1))
    G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    return G, A


def make_sysnet10(num_nodes=1000, avg_deg=6, seed=42, save=True):
    """
    Produces SysNet-10 features:
    ['assets','liabilities','buffer','weights',
     'original_stress','additional_stress',
     'original_losses','additional_losses',
     'original_defaults','additional_defaults']
    """
    np.random.seed(seed)
    G, A = build_graph(num_nodes, avg_deg, seed)
    # Base financial features: sample plausible ranges
    assets = np.random.lognormal(mean=10.0, sigma=0.6, size=(num_nodes, 1)).astype(np.float32)
    liabilities = assets * np.random.uniform(0.6, 0.98, size=(num_nodes, 1)).astype(np.float32)
    buffer = np.random.uniform(0.01, 0.2, size=(num_nodes, 1)).astype(np.float32)
    weights = np.random.uniform(0.0, 1.0, size=(num_nodes, 1)).astype(np.float32)

    original_stress = np.random.beta(a=0.5, b=50.0, size=(num_nodes, 1)).astype(np.float32) * 0.02
    additional_stress = np.random.beta(a=0.3, b=40.0, size=(num_nodes, 1)).astype(np.float32) * 0.02

    original_losses = original_stress * assets * np.random.uniform(0.0001, 0.001, size=(num_nodes, 1)).astype(np.float32)
    additional_losses = additional_stress * assets * np.random.uniform(0.0001, 0.001, size=(num_nodes, 1)).astype(np.float32)

    original_defaults = (original_losses > (assets * 0.001)).astype(np.float32)
    additional_defaults = (additional_losses > (assets * 0.001)).astype(np.float32)

    features = np.hstack([
        assets, liabilities, buffer, weights,
        original_stress, additional_stress,
        original_losses, additional_losses,
        original_defaults, additional_defaults
    ]).astype(np.float32)

    cols = ['assets','liabilities','buffer','weights',
            'original_stress','additional_stress',
            'original_losses','additional_losses',
            'original_defaults','additional_defaults']

    if save:
        np.savez(os.path.join(DATA_DIR, f"symnet10_{num_nodes}.npz"), adjacency=A, features=features)
        pd.DataFrame(features, columns=cols).to_csv(os.path.join(DATA_DIR, f"symnet10_{num_nodes}.csv"), index=False)
    print(f"✅ Generated SysNet-10: nodes={num_nodes}, shape={features.shape}")
    return A, features, cols


def make_sysnet40(num_nodes=1000, avg_deg=6, seed=42, save=True):
    """
    SysNet-40: first 10 are the SysNet-10 features; next 30 are 'extended stress' features
    ext_1 ... ext_30 (stress-like variables)
    """
    A, features10, cols10 = make_sysnet10(num_nodes=num_nodes, avg_deg=avg_deg, seed=seed, save=False)
    np.random.seed(seed + 1)

    # Create 30 extended stress-like features: small betas/gaussians scaled like stress
    ext = (np.random.beta(a=0.4, b=30.0, size=(num_nodes, 30)).astype(np.float32) * 0.02)
    features = np.hstack([features10, ext]).astype(np.float32)
    ext_cols = [f"ext_{i+1}" for i in range(ext.shape[1])]
    cols = cols10 + ext_cols

    if save:
        np.savez(os.path.join(DATA_DIR, f"symnet40_{num_nodes}.npz"), adjacency=A, features=features)
        pd.DataFrame(features, columns=cols).to_csv(os.path.join(DATA_DIR, f"symnet40_{num_nodes}.csv"), index=False)
    print(f"✅ Generated SysNet-40: nodes={num_nodes}, shape={features.shape}")
    return A, features, cols


def load_or_make_eurnet(path=None, num_nodes=250, save=True, seed=42):
    """
    If user provides an EURNET CSV path (rows=banks, cols=features),
    we will load. Otherwise we synthesize a proxy set of banking ratios.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        features = df.values.astype(np.float32)
        # derive adjacency as correlation of features
        adj = np.corrcoef(features.T)
        print(f"✅ Loaded EURNET from {path}, shape={features.shape}")
    else:
        np.random.seed(seed + 10)
        # Synthesize proxy features typical to banking ratios (e.g., ROA, CAR, NPL, LCR, etc.)
        # Create 15 proxy features -> shape (num_nodes, 15)
        f = np.column_stack([
            np.random.normal(0.01, 0.02, size=(num_nodes,)),          # ROA
            np.random.normal(0.12, 0.03, size=(num_nodes,)),          # CAR
            np.random.normal(0.03, 0.01, size=(num_nodes,)),          # NPL ratio
            np.random.normal(0.8, 0.1, size=(num_nodes,)),            # liquidity ratio
            np.random.normal(0.5, 0.15, size=(num_nodes,)),           # leverage
            np.random.normal(1000, 500, size=(num_nodes,)),           # deposits
            np.random.normal(800, 400, size=(num_nodes,)),            # loans
            np.random.normal(0.02, 0.01, size=(num_nodes,)),          # volatility proxy
            np.random.normal(0.04, 0.02, size=(num_nodes,)),          # funding cost
            np.random.normal(0.01, 0.005, size=(num_nodes,)),         # trading pnl ratio
            np.random.normal(0.03, 0.02, size=(num_nodes,)),
            np.random.normal(0.05, 0.02, size=(num_nodes,)),
            np.random.normal(0.02, 0.01, size=(num_nodes,)),
            np.random.normal(1.0, 0.5, size=(num_nodes,)),
            np.random.normal(0.001, 0.001, size=(num_nodes,))
        ]).astype(np.float32)
        features = f
        adj = np.corrcoef(features.T)
        if save:
            np.savez(os.path.join(DATA_DIR, f"eurnet_proxy_{num_nodes}.npz"), adjacency=adj, features=features)
            pd.DataFrame(features).to_csv(os.path.join(DATA_DIR, f"eurnet_proxy_{num_nodes}.csv"), index=False)
        print(f"✅ Generated proxy EURNET: nodes={num_nodes}, shape={features.shape}")
    return adj, features


def inject_anomalies(features, fraction=0.02, magnitude=5.0, seed=1, save=False, base_name=""):
    np.random.seed(seed)
    N = features.shape[0]
    num_anom = max(1, int(N * fraction))
    idx = np.random.choice(N, num_anom, replace=False)
    features2 = features.copy()
    noise = np.abs(np.random.normal(loc=1.0, scale=0.5, size=(num_anom, features.shape[1])))
    features2[idx] += magnitude * noise
    labels = np.zeros(N, dtype=int)
    labels[idx] = 1
    if save:
        np.savetxt(os.path.join(DATA_DIR, f"{base_name}_labels.csv"), labels, fmt="%d", delimiter=",")
        pd.DataFrame(features2).to_csv(os.path.join(DATA_DIR, f"{base_name}_features_injected.csv"), index=False)
    return features2, labels, idx


def augment_and_save(adj, features, base_name, augmentations=("jitter","mode_shift","anomaly"), save=True):
    """
    augmentations:
      - jitter: small gaussian noise copy
      - mode_shift: multiply features by small factor
      - anomaly: inject anomalies (fraction=2%)
    """
    out_files = []
    if "jitter" in augmentations:
        f_jitter = features + np.random.normal(0, 1e-3, features.shape).astype(np.float32)
        if save:
            np.savez(os.path.join(DATA_DIR, f"{base_name}_jitter.npz"), adjacency=adj, features=f_jitter)
            pd.DataFrame(f_jitter).to_csv(os.path.join(DATA_DIR, f"{base_name}_jitter.csv"), index=False)
        out_files.append((adj, f_jitter))
    if "mode_shift" in augmentations:
        shift = 1.02 + np.random.uniform(-0.01, 0.01, size=(features.shape[1],)).astype(np.float32)
        f_shift = features * shift
        if save:
            np.savez(os.path.join(DATA_DIR, f"{base_name}_shift.npz"), adjacency=adj, features=f_shift)
            pd.DataFrame(f_shift).to_csv(os.path.join(DATA_DIR, f"{base_name}_shift.csv"), index=False)
        out_files.append((adj, f_shift))
    if "anomaly" in augmentations:
        f_anom, labels, idx = inject_anomalies(features, fraction=0.02, magnitude=5.0, seed=42, save=save, base_name=base_name)
        if save:
            np.savez(os.path.join(DATA_DIR, f"{base_name}_anomaly.npz"), adjacency=adj, features=f_anom)
        out_files.append((adj, f_anom))
    return out_files


def build_combined(output_path=os.path.join(COMBINED_DIR, "combined_dataset.npz"),
                   include_sys10=True, include_sys40=True, include_eurnet=True):
    """
    Build a unified dataset with FIXED 40-dimensional feature vectors.
    SysNet-10  -> padded to 40 dims
    SysNet-40  -> already 40 dims
    EURNET(15) -> padded to 40 dims
    """

    TARGET_DIM = 40
    datasets = []

    # ---- Load originals ----
    if include_sys10:
        A10, F10, _ = make_sysnet10(num_nodes=1000, save=True)
        # pad SysNet-10 → (1000, 40)
        pad_10 = np.zeros((F10.shape[0], TARGET_DIM - F10.shape[1]), dtype=np.float32)
        F10_padded = np.hstack([F10, pad_10])
        datasets.append((A10, F10_padded, "sysnet10"))

    if include_sys40:
        A40, F40, _ = make_sysnet40(num_nodes=1000, save=True)
        # already (1000,40)
        datasets.append((A40, F40, "sysnet40"))

    if include_eurnet:
        Ae, Fe = load_or_make_eurnet(None, num_nodes=250, save=True)
        # pad EURNET (250,15) → (250,40)
        pad_e = np.zeros((Fe.shape[0], TARGET_DIM - Fe.shape[1]), dtype=np.float32)
        Fe_padded = np.hstack([Fe, pad_e])
        datasets.append((Ae, Fe_padded, "eurnet"))

    # ---- Augment & collect ----
    all_features = []
    all_adjs = []

    for (A, F, name) in datasets:
        # add original
        all_features.append(F)
        all_adjs.append(A)

        # augmented sets
        aug = augment_and_save(A, F, name, augmentations=("jitter","mode_shift","anomaly"), save=True)
        for aA, aF in aug:
            all_features.append(aF)
            all_adjs.append(aA)

    # ---- Combine features (now all 40-dim) ----
    features_stack = np.vstack(all_features).astype(np.float32)

    # ---- Block diagonal adjacency (keep datasets separate) ----
    total_nodes = sum(a.shape[0] for a in all_adjs)
    bigA = np.zeros((total_nodes, total_nodes), dtype=np.float32)

    start = 0
    for a in all_adjs:
        n = a.shape[0]
        bigA[start:start+n, start:start+n] = a
        start += n

    # ---- Save ----
    np.savez(output_path, adjacency=bigA, features=features_stack)
    pd.DataFrame(features_stack).to_csv(output_path.replace(".npz", ".csv"), index=False)

    print("---------------------------------------------------------")
    print(f"✅ Combined dataset saved → {output_path}")
    print(f"📌 Total nodes      : {features_stack.shape[0]}")
    print(f"📌 Feature dimension: {features_stack.shape[1]}  (fixed 40)")
    print("---------------------------------------------------------")

    return bigA, features_stack


def main(args):
    # build originals and combined
    build_combined()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eurnet", type=str, default=None, help="path to eurnet csv (optional)")
    args = parser.parse_args()
    main(args)
