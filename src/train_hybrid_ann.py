# src/train_hybrid_ann.py
"""
Training script (upgraded) for Hybrid ANN + GNN + Autoencoder.
Key features:
 - Focal loss for classification head to handle extreme imbalance
 - Small centroid contrastive loss (keeps embeddings separable)
 - Per-epoch validation split (node-wise) to compute best threshold by F1
 - Save best checkpoint by validation F1 (and write threshold to experiments/thresholds.txt)
 - Lightweight, runs on CPU (but uses GPU if available)
"""

import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from utils import normalize_adj, to_torch_tensor
from data_loader import generate_symnet_large, inject_synthetic_anomalies
from model_hybrid_ann import HybridGNN_AE_ANN

# simple focal loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=5.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal = (self.alpha * (1 - p_t) ** self.gamma) * bce
        return focal.mean()

def load_data(num_nodes=1000, inject_frac=0.02, seed=42):
    base = f"data/symnet_data_{num_nodes}"
    npz = base + ".npz"
    labels_file = base + "_labels.csv"
    feat_file = base + "_features_injected.csv"

    if os.path.exists(npz) and os.path.exists(labels_file) and os.path.exists(feat_file):
        import pandas as pd
        adj = np.load(npz)["adjacency"]
        features = pd.read_csv(feat_file).values.astype('float32')
        labels = np.loadtxt(labels_file, dtype=int, delimiter=',')
    else:
        adj, features = generate_symnet_large(num_nodes=num_nodes, feature_dim=6, random_seed=seed, save=True)
        features, labels, _ = inject_synthetic_anomalies(adj, features, anomaly_fraction=inject_frac, seed=seed, save=True)
    return adj, features, labels

def centroid_contrastive_loss(Z, labels, margin=1.0):
    device = Z.device
    labels_t = torch.tensor(labels, device=device)
    if labels_t.sum() == 0 or (labels_t == 0).sum() == 0:
        return torch.tensor(0.0, device=device)
    z_norm = Z[labels_t == 0].mean(dim=0)
    z_anom = Z[labels_t == 1].mean(dim=0)
    dist = torch.norm(z_norm - z_anom)
    loss = torch.relu(margin - dist)
    return loss

def compute_best_threshold_and_f1(y_true, probs):
    # search thresholds from 0.0 to 1.0 in fine grid, pick threshold maximizing F1
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, 201):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def train(epochs=60, lr=1e-3, num_nodes=1000, inject_frac=0.02, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    adj, features, labels = load_data(num_nodes=num_nodes, inject_frac=inject_frac, seed=seed)
    A_norm = normalize_adj(adj)
    X_full = to_torch_tensor(features).to(device)
    A_t = to_torch_tensor(A_norm).to(device)
    y = labels.copy()

    # make node-wise train/val split (stratified by label if possible)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    anom_idx = idx[labels == 1]
    norm_idx = idx[labels == 0]
    rng.shuffle(anom_idx); rng.shuffle(norm_idx)

    # keep about 20% for val (but ensure >=1 anomaly in val)
    val_frac = 0.2
    n_val_anom = max(1, int(len(anom_idx) * val_frac))
    n_val_norm = int(len(norm_idx) * val_frac)

    val_idx = np.concatenate([anom_idx[:n_val_anom], norm_idx[:n_val_norm]])
    train_idx = np.setdiff1d(idx, val_idx)

    print(f"Total nodes: {len(labels)}, Train: {len(train_idx)}, Val: {len(val_idx)} (Anom in val: {n_val_anom})")

    # create datasets tensors (we will index X_full by node idx during training)
    X_train = X_full[train_idx]
    X_val = X_full[val_idx]
    A_train = A_t  # full adjacency used for GNN (we keep graph whole; learning uses nodes' features)
    # note: training uses full A but only measures val metrics on val_idx nodes

    model = HybridGNN_AE_ANN(in_dim=X_full.shape[1],
                             ann_hidden1=64, ann_hidden2=64,
                             gnn_hidden=128, embed_dim=64, assist_dim=32, decoder_hidden=64).to(device)

    recon_fn = torch.nn.MSELoss()
    cls_fn = FocalLoss(alpha=5.0, gamma=3.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)

    best_val_f1 = -1.0
    best_ckpt = None
    best_threshold = 0.5

    for ep in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        # forward on full graph (so GNN sees all nodes' connectivity)
        X_recon, Z, logits, X_ann = model(X_full, A_t)
        # recon loss on full nodes
       # Weighted reconstruction (important for anomaly sensitivity)
        feature_weights = torch.tensor([1,1,1,1,2,2], dtype=torch.float32, device=device)  # last 2 stress features weighted more
        recon_loss = torch.mean(feature_weights * (X_recon - X_full)**2)


        # classification loss only computed on train_idx nodes (so val nodes are unseen for training)
        logits_train = logits[train_idx]
        labels_train = torch.tensor(labels[train_idx], dtype=torch.float32, device=device)
        cls_loss = cls_fn(logits_train, labels_train)

        # centroid contrastive on full embeddings (cheap)
        ctr_loss = centroid_contrastive_loss(Z, labels)

        total_loss = recon_loss + 1.5 * cls_loss + 0.5 * ctr_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        # Validation eval (compute probs on val nodes)
        model.eval()
        with torch.no_grad():
            _, Z_val, logits_all, _ = model(X_full, A_t)
            probs_all = torch.sigmoid(logits_all).cpu().numpy()
            probs_val = probs_all[val_idx]
            y_val = labels[val_idx]

            # choose best threshold on val by F1
            t_val, f1_val = compute_best_threshold_and_f1(y_val, probs_val)
            preds_val = (probs_val >= t_val).astype(int)
            prec_val = precision_score(y_val, preds_val, zero_division=0)
            rec_val = recall_score(y_val, preds_val, zero_division=0)

            # compute AUC too
            try:
                auc_val = roc_auc_score(y_val, probs_val)
            except:
                auc_val = float('nan')

        # Logging
        if ep % 5 == 0 or ep == 1:
            print(f"Epoch [{ep}/{epochs}] Loss={total_loss.item():.4f} Recon={recon_loss.item():.4f} Cls={cls_loss.item():.4f} Ctr={ctr_loss.item():.4f} VAL_F1={f1_val:.4f} VAL_T={t_val:.3f} VAL_AUC={auc_val:.4f}")

        # checkpoint by val F1
        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_threshold = t_val
            ckpt = {
                'model_state': model.state_dict(),
                'best_val_f1': best_val_f1,
                'best_threshold': best_threshold,
                'epoch': ep
            }
            torch.save(ckpt, "experiments/hybrid_ann_model_best_byF1.pt")
            # also write threshold to text for evaluator
            with open("experiments/thresholds.txt", "w") as f:
                f.write(str(best_threshold))

    # final save (last model)
    torch.save({'model_state': model.state_dict()}, "experiments/hybrid_ann_model.pt")
    print(f"Training done. Best val F1={best_val_f1:.4f} at threshold={best_threshold:.4f}")
    print("Saved best model to experiments/hybrid_ann_model_best_byF1.pt and experiments/hybrid_ann_model.pt")


if __name__ == "__main__":
    train()
