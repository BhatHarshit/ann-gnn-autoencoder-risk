# src/model_ann_baseline.py

import torch
import torch.nn as nn

class ANNBaseline(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)  # logits
        )
    def forward(self, x):
        return self.net(x).squeeze(1)
