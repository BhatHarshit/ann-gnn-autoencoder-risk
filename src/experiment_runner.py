# src/experiment_runner.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.serialization
import matplotlib.pyplot as plt  # <-- Added for plotting

# -------------------------------------------------------
# FIXED — universal modern-safe numpy scalar registration
# -------------------------------------------------------
torch.serialization.add_safe_globals([
    np.dtype,
    np.generic,   # replaces deprecated multiarray.scalar safely for ALL numpy versions
])

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split

from data_loader import generate_symnet_large, inject_synthetic_anomalies
from utils import normalize_adj, to_torch_tensor

# Models
from model_ann_baseline import ANNBaseline
from model_autoencoder import AutoEncoder
from model_gnn import GNNEncoder
from model_gnn_autoencoder import GNNAutoEncoder
from model_hybrid_ann import HybridGNN_AE_ANN

# -------------------------
# dataset loader
# -------------------------
def load_dataset(num_nodes=1000):
    base = f"data/symnet_data_{num_nodes}"
    npz_path = base + ".npz"
    labels_path = base + "_labels.csv"
    features_injected_path = base + "_features_injected.csv"

    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        adj = npz["adjacency"]
    else:
        adj, _ = generate_symnet_large(num_nodes=num_nodes)

    if os.path.exists(features_injected_path) and os.path.exists(labels_path):
        features = pd.read_csv(features_injected_path).values.astype("float32")
        labels = np.loadtxt(labels_path, dtype=int, delimiter=",")
    else:
        adj, features = generate_symnet_large(num_nodes=num_nodes)
        features, labels, _ = inject_synthetic_anomalies(adj, features)

    return adj.astype("float32"), features.astype("float32"), labels.astype(int)

# -------------------------
# metrics helper
# -------------------------
def evaluate_model(labels, probs, preds, recon_err=None):
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": int(cm[1, 1]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TN": int(cm[0, 0]),
        "ReconErrMean": float(np.mean(recon_err)) if recon_err is not None else None
    }

# -------------------------
# tolerant state_dict loader
# -------------------------
def flexible_load_state_dict(model, ckpt_state):
    model_state = model.state_dict()
    new_state = {}
    loaded_keys = []
    skipped_keys = []
    mismatched = []

    for ck_k, ck_v in ckpt_state.items():
        candidates = [ck_k]
        if ".net." in ck_k:
            candidates.append(ck_k.replace(".net.", "."))
        if ".module." in ck_k:
            candidates.append(ck_k.replace(".module.", "."))

        matched = False
        for cand in candidates:
            if cand in model_state:
                if list(model_state[cand].shape) == list(ck_v.shape):
                    new_state[cand] = ck_v
                    loaded_keys.append(cand)
                    matched = True
                    break
                else:
                    mismatched.append((cand, tuple(model_state[cand].shape), tuple(ck_v.shape)))
                    matched = True
                    break

        if not matched:
            skipped_keys.append(ck_k)

    print(f"Flexible loader: {len(loaded_keys)} loaded, {len(mismatched)} mismatched, {len(skipped_keys)} skipped.")
    model.load_state_dict({**model_state, **new_state}, strict=False)
    return loaded_keys, mismatched, skipped_keys

# -------------------------
# THRESHOLD SEARCH HELPERS
# -------------------------
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

def find_best_threshold(labels, probs, method="f1", num_steps=1001, return_metric=False, required_recall=0.9):
    labels = np.asarray(labels)
    probs = np.asarray(probs).ravel()

    if method == "youden":
        fpr, tpr, th = roc_curve(labels, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_t = float(th[best_idx])
        best_val = float(j_scores[best_idx])
        return (best_t, best_val) if return_metric else best_t

    if method == "precision_at_recall":
        prec, rec, th = precision_recall_curve(labels, probs)
        idxs = np.where(rec >= required_recall)[0]
        if len(idxs) == 0:
            method = "f1"
        else:
            chosen = idxs[np.argmax(prec[idxs])]
            if chosen >= len(th):
                best_t = 1.0
            else:
                best_t = float(th[chosen])
            best_val = float(prec[chosen])
            return (best_t, best_val) if return_metric else best_t

    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.0, 1.0, num_steps):
        preds = (probs >= t).astype(int)
        s = f1_score(labels, preds, zero_division=0)
        if s > best_score:
            best_score = float(s)
            best_t = float(t)
    return (best_t, best_score) if return_metric else best_t

# -------------------------
# PLOTTING / VISUALIZATION FUNCTIONS
# -------------------------
def plot_reconstruction_error(recon_err, labels, out_path="experiments/compare/recon_error_plot.png"):
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(recon_err)), recon_err, c=labels, cmap="coolwarm", alpha=0.6)
    plt.colorbar(label="True label")
    plt.xlabel("Node index")
    plt.ylabel("Reconstruction error")
    plt.title("Hybrid Model Reconstruction Error")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Reconstruction error plot saved to {out_path}")

def plot_probs_histogram(probs, labels, out_path="experiments/compare/probs_histogram.png"):
    plt.figure(figsize=(8, 4))
    plt.hist([probs[labels==0], probs[labels==1]], bins=50, label=["Normal", "Anomaly"], color=["green", "red"], alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Hybrid Model Predicted Probabilities")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Probability histogram saved to {out_path}")

# -------------------------
# Optional 1: Additional Hybrid Plots (Optional Work 1)
# -------------------------
def plot_hybrid_latent_space(Z, labels, out_path="experiments/compare/hybrid_latent_space.png"):
    """Plot 2D latent embeddings colored by labels."""
    if Z.shape[1] > 2:
        from sklearn.decomposition import PCA
        Z_2d = PCA(n_components=2).fit_transform(Z.cpu().numpy())
    else:
        Z_2d = Z.cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap="coolwarm", alpha=0.6)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("Hybrid Model Latent Space")
    plt.colorbar(label="True label")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Latent space plot saved to {out_path}")

# -------------------------
# Optional 3: ROC and PR Curves (Optional Work 3)
# -------------------------
def plot_roc_pr_curves(labels, probs, out_root="experiments/compare"):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc_val = auc(rec, prec)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc_val:.4f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_root, "hybrid_roc_curve.png"), dpi=150)
    plt.close()
    print(f"📊 ROC curve saved to {os.path.join(out_root, 'hybrid_roc_curve.png')}")

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=f"PR curve (AUC={pr_auc_val:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(out_root, "hybrid_pr_curve.png"), dpi=150)
    plt.close()
    print(f"📊 PR curve saved to {os.path.join(out_root, 'hybrid_pr_curve.png')}")

# -------------------------
# Optional 4: Confusion Matrix Heatmap (Optional Work 4)
# -------------------------
def plot_confusion_matrix_heatmap(cm, out_path="experiments/compare/hybrid_confusion_matrix.png"):
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix heatmap saved to {out_path}")

# -------------------------
# main experiments runner
# -------------------------
def run_all_experiments(num_nodes=1000, out_root="experiments/compare"):
    print("📌 Loading dataset...")
    os.makedirs(out_root, exist_ok=True)

    adj, features, labels = load_dataset(num_nodes)
    A_norm = normalize_adj(adj)

    # -------------------------
    # 70/15/15 split
    # -------------------------
    idx = np.arange(len(labels))
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=42, stratify=labels)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, stratify=labels[idx_temp])

    X = to_torch_tensor(features).float()
    A = to_torch_tensor(A_norm).float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    A = A.to(device)

    results = []
    val_metrics = {}
    test_metrics = {}
    in_dim = X.shape[1]

    # -------------------------
    # 1) ANN baseline
    # -------------------------
    print("\n⚡ Running ANN baseline...")
    model = ANNBaseline(in_dim=in_dim).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    m = evaluate_model(labels, probs, preds)
    m["Model"] = "ANN"
    results.append(m)

    # -------------------------
    # 2) AutoEncoder baseline
    # -------------------------
    print("\n⚡ Running AutoEncoder baseline...")
    model = AutoEncoder(input_dim=in_dim).to(device)
    model.eval()
    with torch.no_grad():
        out = model(X)
    X_recon = out[0] if isinstance(out, tuple) else out
    recon_err = torch.mean((X - X_recon)**2, dim=1).cpu().numpy()
    ae_probs = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-12)
    ae_preds = (ae_probs >= 0.5).astype(int)
    m = evaluate_model(labels, ae_probs, ae_preds, recon_err)
    m["Model"] = "AutoEncoder"
    results.append(m)

    # -------------------------
    # 3) GNN baseline
    # -------------------------
    print("\n⚡ Running GNN baseline...")
    model = GNNEncoder(in_dim=in_dim, hidden_dim=64, embed_dim=32).to(device)
    model.eval()
    with torch.no_grad():
        Z = model(X, A)
        logits = torch.sum(Z, dim=1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    m = evaluate_model(labels, probs, preds)
    m["Model"] = "GNN"
    results.append(m)

    # -------------------------
    # 4) GNN + Autoencoder
    # -------------------------
    print("\n⚡ Running GNN + Autoencoder...")
    model = GNNAutoEncoder(in_dim=in_dim, gnn_hidden=64, embed_dim=32).to(device)
    model.eval()
    with torch.no_grad():
        X_recon, Z = model(X, A)
        recon_err = torch.mean((X - X_recon)**2, dim=1).cpu().numpy()
    probs = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-12)
    preds = (probs >= 0.5).astype(int)
    m = evaluate_model(labels, probs, preds, recon_err)
    m["Model"] = "GNN+AE"
    results.append(m)

    # -------------------------
    # 5) Hybrid ANN + GNN + AE
    # -------------------------
    print("\n⚡ Running Hybrid ANN+GNN+AE...")
    model = HybridGNN_AE_ANN(in_dim=in_dim).to(device)

    ckpt_path = "experiments/hybrid_ann_model_best_byF1.pt"
    best_threshold = 0.5
    if os.path.exists(ckpt_path):
        print(f"🔄 Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_state = ckpt.get("model_state", ckpt)
        loaded, mismatched, skipped = flexible_load_state_dict(model, ckpt_state)
        print(f"→ Loaded {len(loaded)} weights.")
        if isinstance(ckpt, dict):
            best_threshold = ckpt.get("best_threshold",
                               ckpt.get("best_thresh",
                               ckpt.get("threshold",
                               ckpt.get("best_threshold_value", best_threshold))))
    else:
        print("⚠ No checkpoint found, evaluating untrained model.")

    model.eval()
    with torch.no_grad():
        out = model(X, A)
        if isinstance(out, tuple) and len(out) == 4:
            X_recon, Z, logits, X_ann = out
        elif isinstance(out, tuple) and len(out) == 3:
            X_recon, Z, logits = out
            X_ann = None
        else:
            raise ValueError("Hybrid model returned unexpected output format")

    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze()
    probs = torch.sigmoid(logits).cpu().numpy()
    recon_err = torch.mean((X - X_recon)**2, dim=1).cpu().numpy()

    # -------------------------
    # Compute threshold on VAL set only
    # -------------------------
    try:
        thr, thr_score = find_best_threshold(labels[idx_val], probs[idx_val], method="f1", num_steps=1001, return_metric=True)
        best_threshold = float(thr)
        print(f"→ Best threshold (from VAL) : {best_threshold:.4f}  (F1={thr_score:.4f})")
    except Exception as e:
        print("→ VAL Threshold search failed, falling back to:", best_threshold, "Error:", e)

    try:
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "hybrid_best_threshold.txt"), "w") as fh:
            fh.write(f"{best_threshold}\n")
    except Exception:
        pass

    # -------------------------
    # VAL metrics
    # -------------------------
    val_preds = (probs[idx_val] >= best_threshold).astype(int)
    val_metrics = evaluate_model(labels[idx_val], probs[idx_val], val_preds, recon_err[idx_val])
    val_metrics["Model"] = "Hybrid_ANN_GNN_AE"

    # -------------------------
    # TEST metrics
    # -------------------------
    test_preds = (probs[idx_test] >= best_threshold).astype(int)
    test_metrics = evaluate_model(labels[idx_test], probs[idx_test], test_preds, recon_err[idx_test])
    test_metrics["Model"] = "Hybrid_ANN_GNN_AE"

    # -------------------------
    # Append Hybrid to results (test set)
    # -------------------------
    results.append(test_metrics)

    # -------------------------
    # Save JSON metrics
    # -------------------------
    try:
        with open(os.path.join(out_root, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=4)
        with open(os.path.join(out_root, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)
    except Exception:
        pass

    # -------------------------
    # Save comparison table CSV
    # -------------------------
    df = pd.DataFrame(results)
    os.makedirs(out_root, exist_ok=True)
    df.to_csv(os.path.join(out_root, "comparison_table.csv"), index=False)
    print("\n📁 Saved:", os.path.join(out_root, "comparison_table.csv"))

    # -------------------------
    # PLOT RECONSTRUCTION ERRORS AND PROBABILITIES
    # -------------------------
    plot_reconstruction_error(recon_err, labels, os.path.join(out_root, "hybrid_recon_error.png"))
    plot_probs_histogram(probs, labels, os.path.join(out_root, "hybrid_probs_histogram.png"))

    # -------------------------
    # Optional plots (1,3,4)
    # -------------------------
    plot_hybrid_latent_space(Z, labels, os.path.join(out_root, "hybrid_latent_space.png"))
    plot_roc_pr_curves(labels, probs, out_root)
    plot_confusion_matrix_heatmap(confusion_matrix(labels[idx_test], test_preds), os.path.join(out_root, "hybrid_confusion_matrix.png"))

    return df


if __name__ == "__main__":
    df = run_all_experiments()
    print(df)
