# src/evaluate_and_visualize_ann_full.py
"""
Final Evaluation Script for Hybrid ANN + GNN + Autoencoder Model
Includes:
 - Robust checkpoint loading
 - Auto-threshold selection (Youden’s J)
 - Probability histogram
 - Full visualization suite
 - Handles older/newer variations of model output signatures
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.serialization

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, auc
)
from sklearn.decomposition import PCA

from data_loader import generate_symnet_large, inject_synthetic_anomalies
from utils import normalize_adj, to_torch_tensor
from model_hybrid_ann import HybridGNN_AE_ANN

# Fix PyTorch safety for numpy reconstruct
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


# ------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------
def load_data(num_nodes=1000):
    base = f"data/symnet_data_{num_nodes}"
    npz_path = base + ".npz"
    labels_path = base + "_labels.csv"
    features_injected_path = base + "_features_injected.csv"

    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        adj = npz["adjacency"]
    else:
        adj, _ = generate_symnet_large(num_nodes=num_nodes)

    # Load injected features if exist
    if os.path.exists(features_injected_path) and os.path.exists(labels_path):
        import pandas as pd
        features = pd.read_csv(features_injected_path).values.astype("float32")
        labels = np.loadtxt(labels_path, dtype=int, delimiter=",")
    else:
        adj, features = generate_symnet_large(num_nodes=num_nodes)
        features, labels, _ = inject_synthetic_anomalies(adj, features)

    return adj, features, labels


# ------------------------------------------------------
# Threshold Finder — OPTION A (Youden's J)
# ------------------------------------------------------
def compute_optimal_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_t = thresholds[best_idx]

    print(f"📌 AUTO-SELECTED optimal threshold = {best_t:.4f}")
    return best_t


# ------------------------------------------------------
# CSV SAVE
# ------------------------------------------------------
def save_csv(arr, path, fmt="%.6f"):
    np.savetxt(path, arr, delimiter=",", fmt=fmt)


# ------------------------------------------------------
# ROBUST CHECKPOINT LOADER
# ------------------------------------------------------
def _load_checkpoint_to_model(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    return ckpt


# ------------------------------------------------------
# MAIN EVALUATION PIPELINE
# ------------------------------------------------------
def evaluate_and_visualize(model_path="experiments/hybrid_ann_model.pt",
                           num_nodes=1000,
                           out_dir="experiments/results"):

    print("📌 Loading dataset...")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    adj, features, labels = load_data(num_nodes=num_nodes)
    A_norm = normalize_adj(adj)
    X = to_torch_tensor(features).to(device)
    A_t = to_torch_tensor(A_norm).to(device)

    print(f"📌 Loading checkpoint from: {model_path}")

    # Instantiate model with TRAINING architecture
    model = HybridGNN_AE_ANN(
        in_dim=X.shape[1],
        ann_hidden1=64,
        ann_hidden2=64,
        gnn_hidden=128,
        embed_dim=64,
        assist_dim=32,
        decoder_hidden=64
    ).to(device)

    # Robust loading
    ckpt = _load_checkpoint_to_model(model, model_path, device)
    model.eval()

    # Inference
    with torch.no_grad():
        out = model(X, A_t)

        if len(out) == 4:
            X_recon, Z, logits, X_ann = out
        elif len(out) == 3:
            X_recon, Z, logits = out
            X_ann = None
        else:
            raise ValueError(f"Unexpected model output len: {len(out)}")

    # Convert logits → probability
    probs_np = torch.sigmoid(logits).cpu().numpy()
    recon_err = torch.mean((X - X_recon) ** 2, dim=1).cpu().numpy()

    # AUTO THRESHOLD
    best_t = compute_optimal_threshold(labels, probs_np)
    preds = (probs_np >= best_t).astype(int)

    # Save CSV files
    save_csv(recon_err, os.path.join(out_dir, "anomaly_scores.csv"))
    save_csv(probs_np, os.path.join(out_dir, "predicted_probabilities.csv"))
    save_csv(preds, os.path.join(out_dir, "predictions.csv"), fmt="%d")
    save_csv(labels, os.path.join(out_dir, "labels.csv"), fmt="%d")

    # Metrics
    roc_auc = roc_auc_score(labels, probs_np)
    precision, recall, pr_thr = precision_recall_curve(labels, probs_np)
    avg_precision = average_precision_score(labels, probs_np)

    cm = confusion_matrix(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    # Print results
    print("✅ Evaluation Complete")
    print(f"Mean Reconstruction Error: {recon_err.mean():.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"AP Score: {avg_precision:.4f}")
    print(f"Best Threshold: {best_t:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision/Recall/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}")

    # Save metrics summary
    with open(os.path.join(out_dir, "metrics_summary.txt"), "w") as f:
        f.write(f"Mean_recon_err,{recon_err.mean():.6f}\n")
        f.write(f"ROC_AUC,{roc_auc:.6f}\n")
        f.write(f"AP,{avg_precision:.6f}\n")
        f.write(f"BestThreshold,{best_t:.6f}\n")
        f.write(f"Precision,{prec:.6f}\n")
        f.write(f"Recall,{rec:.6f}\n")
        f.write(f"F1,{f1:.6f}\n")

    # --------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------

    # Probability Histogram
    plt.figure()
    plt.hist(probs_np, bins=40)
    plt.title("Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, "probability_histogram.png"))
    plt.close()

    # (All your other plots preserved exactly)
    # Adjacency, degree distribution, PCA, ROC, PR, confusion matrix, graph coloring…

    # 1. Adjacency Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(adj, cmap="hot")
    plt.title("Adjacency Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, "adjacency_heatmap.png"))
    plt.close()

    # 2. Degree Distribution
    degs = adj.sum(axis=0)
    plt.figure()
    plt.hist(degs, bins=40)
    plt.title("Degree Distribution")
    plt.savefig(os.path.join(out_dir, "degree_distribution.png"))
    plt.close()

    # 3–4. PCA
    if Z is not None:
        Z_np = Z.cpu().numpy()
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z_np)

        plt.figure(figsize=(6, 5))
        plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="coolwarm", s=12)
        plt.title("PCA — True Labels")
        plt.savefig(os.path.join(out_dir, "pca_true.png"))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(Z2[:, 0], Z2[:, 1], c=preds, cmap="coolwarm", s=12)
        plt.title("PCA — Predicted Labels")
        plt.savefig(os.path.join(out_dir, "pca_pred.png"))
        plt.close()

    # 5. Sorted reconstruction error plot
    idx = np.argsort(recon_err)
    plt.figure(figsize=(8, 3))
    plt.plot(recon_err[idx], ".", markersize=3)
    plt.title("Sorted Reconstruction Error")
    plt.savefig(os.path.join(out_dir, "recon_error_sorted.png"))
    plt.close()

    # 6. ROC curve
    fpr, tpr, _ = roc_curve(labels, probs_np)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # 7. Precision–Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f"AP={avg_precision:.3f}")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()

    # 8. Confusion Matrix Image
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # 9. Network Graph with TN/TP/FP/FN Colors
    G = nx.from_numpy_array(adj)
    colors = []
    for i in range(len(labels)):
        t = labels[i]
        p = preds[i]
        if t == 0 and p == 0: colors.append("skyblue")
        elif t == 1 and p == 1: colors.append("red")
        elif t == 0 and p == 1: colors.append("orange")
        else: colors.append("purple")

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=colors, node_size=20, edge_color="gray")
    plt.title("Graph — TN/TP/FP/FN")
    plt.savefig(os.path.join(out_dir, "graph_confusion.png"))
    plt.close()

    print("\n✅ All evaluation outputs saved to:", out_dir)


if __name__ == "__main__":
    evaluate_and_visualize()
