# src/model_hybrid_ann.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Simple 2-layer GCN used in the trained checkpoint
# ---------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        AX = torch.matmul(A, X)
        return F.relu(self.linear(AX))


# ---------------------------------------------------------
# Final Hybrid Model (Checkpoint-compatible version)
# ---------------------------------------------------------
class HybridGNN_AE_ANN(nn.Module):
    """
    Hybrid ANN + GNN + Assist + Decoder + Classifier

    Checkpoint-compatible: preserves module names & layer indices used by your saved checkpoint.
    Additions:
      - recon_proj: projects normalized reconstruction error (scalar) -> 64-d and is added
                    to fusion features before classification (non-destructive to shapes).
      - logit_scale: learnable scalar used to scale logits before sigmoid (temperature-like).
    """

    def __init__(self, in_dim=6):
        super().__init__()

        # -------------------------
        # 1) ANN Feature Transformer (index alignment preserved)
        # -------------------------
        self.ann_feat = nn.Module()
        self.ann_feat.net = nn.Sequential(
            nn.Linear(in_dim, 64),    # ann_feat.net.0.*
            nn.ReLU(),
            nn.Dropout(p=0.1),        # keep an extra op to match index 3 for next Linear
            nn.Linear(64, 64),        # ann_feat.net.3.*
            nn.ReLU(),
        )

        # -------------------------
        # 2) GNN Encoder (64 -> 128 -> 64)
        # names: gnn.layer1.linear.*, gnn.layer2.linear.*
        # -------------------------
        self.gnn = nn.Module()
        self.gnn.layer1 = GCNLayer(64, 128)
        self.gnn.layer2 = GCNLayer(128, 64)

        # -------------------------
        # 3) ANN Assist Branch (Linear -> ReLU -> BatchNorm1d)
        # -------------------------
        self.ann_assist = nn.Sequential(
            nn.Linear(64, 32),     # ann_assist.0.weight
            nn.ReLU(),
            nn.BatchNorm1d(32),    # ann_assist.2.*
        )

        # -------------------------
        # 4) Fusion block (Linear(96,64) -> ReLU -> BatchNorm1d(64))
        # -------------------------
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),    # fusion.0.weight
            nn.ReLU(),
            nn.BatchNorm1d(64),   # fusion.2.*
        )

        # -------------------------
        # 5) Decoder (64 -> 64 -> in_dim)
        # -------------------------
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),    # decoder.0.*
            nn.ReLU(),
            nn.Linear(64, in_dim) # decoder.2.*
        )

        # -------------------------
        # 6) Classifier (64 -> 32 -> 1) (BatchNorm retained)
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),    # classifier.0.*
            nn.ReLU(),
            nn.BatchNorm1d(32),   # classifier.2.*
            nn.Linear(32, 1),     # classifier.3.*
        )

        # -------------------------
        # ADDITIONS (non-breaking)
        # - recon_proj: project scalar normalized recon error -> 64-d vector
        # - logit_scale: learnable scalar to rescale logits (temperature-like)
        # -------------------------
        # placed as new attributes so they won't clash with checkpoint keys
        self.recon_proj = nn.Linear(1, 64)   # maps recon_norm -> 64
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # multiply logits by this before sigmoid

        # small init for recon_proj to start near-zero (so old behavior preserved initially)
        nn.init.zeros_(self.recon_proj.weight)
        nn.init.zeros_(self.recon_proj.bias)

    def forward(self, X, A):
        # 1. ANN feature transform (ann_feat.net)
        if hasattr(self.ann_feat, "net"):
            X_ann = self.ann_feat.net(X)   # [N, 64]
        else:
            X_ann = self.ann_feat(X)

        # 2. GNN embedding (uses X_ann as node features)
        Z1 = self.gnn.layer1(X_ann, A)     # [N,128]
        Z = self.gnn.layer2(Z1, A)         # [N,64]

        # 3. Assist branch runs on Z
        assist = self.ann_assist(Z)        # [N,32]

        # 4. Fusion: concat X_ann (64) + assist (32) => 96
        fusion_input = torch.cat([X_ann, assist], dim=1)  # [N,96]
        F = self.fusion(fusion_input)      # [N,64]

        # 5. Reconstruction (decoder)
        X_recon = self.decoder(F)          # [N, in_dim]

        # 6a. Compute reconstruction error scalar per-node (MSE)
        # Use float ops; will be used to inform classifier via recon_proj
        recon_err = torch.mean((X - X_recon) ** 2, dim=1, keepdim=True)  # [N,1]

        # Normalize recon_err across batch (small eps for safety) — stable for inference
        re_min = recon_err.min(dim=0, keepdim=True)[0]
        re_max = recon_err.max(dim=0, keepdim=True)[0]
        denom = (re_max - re_min).clamp_min(1e-12)
        recon_norm = (recon_err - re_min) / denom  # [N,1]

        # 6b. Project recon_norm -> 64-d and add to fused features (this lets classifier use AE signal)
        recon_feat = self.recon_proj(recon_norm)   # [N,64]
        F_comb = F + recon_feat                    # [N,64]  (elementwise add preserves shape)

        # 7. Classifier logits (apply classifier to combined features)
        logits = self.classifier(F_comb).squeeze(1)  # [N]

        # 8. Apply learnable logit scaling before sigmoid (temperature-like)
        logits = logits * self.logit_scale

        return X_recon, Z, logits, X_ann

    # -------------------------
    # Optional helper: compute novelty score
    # -------------------------
    def novelty_score(self, X, A):
        """
        Returns probs, recon_err, novelty_score:
          novelty = normalized_recon_err * prob
        Useful for ranking / paper plots.
        """
        self.eval()
        with torch.no_grad():
            X_recon, Z, logits, X_ann = self.forward(X, A)
            probs = torch.sigmoid(logits)
            recon_err = torch.mean((X - X_recon) ** 2, dim=1)

            # normalize recon errs 0-1
            re_min = recon_err.min()
            re_max = recon_err.max()
            if (re_max - re_min) > 1e-12:
                recon_norm = (recon_err - re_min) / (re_max - re_min)
            else:
                recon_norm = torch.zeros_like(recon_err)

            novelty = recon_norm * probs

        return probs.cpu(), recon_err.cpu(), novelty.cpu()
