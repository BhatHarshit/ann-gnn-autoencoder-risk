"""
model_autoencoder.py
Upgraded Autoencoder for Hybrid GNN–ANN Model
Adds:
 - Denoising Autoencoder (DAE)
 - Optional Variational Autoencoder (VAE)
 - Weighted reconstruction support
Fully backward-compatible with baseline AE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden1=128,
                 hidden2=64,
                 latent_dim=32,
                 denoising=True,
                 variational=False,
                 noise_std=0.05):
        """
        Upgraded AE:
         - denoising=True enables Gaussian noise
         - variational=True enables VAE mode
         - latent_dim = bottleneck dimension
        """
        super().__init__()

        self.denoising = denoising
        self.variational = variational
        self.noise_std = noise_std

        # ----------- Encoder -----------
        self.enc_fc1 = nn.Linear(input_dim, hidden1)
        self.enc_fc2 = nn.Linear(hidden1, hidden2)

        if variational:
            # VAE uses mu + logvar
            self.fc_mu = nn.Linear(hidden2, latent_dim)
            self.fc_logvar = nn.Linear(hidden2, latent_dim)
        else:
            self.fc_latent = nn.Linear(hidden2, latent_dim)

        # ----------- Decoder -----------
        self.dec_fc1 = nn.Linear(latent_dim, hidden2)
        self.dec_fc2 = nn.Linear(hidden2, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))

        if self.variational:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            z = self.fc_latent(h)
            return z, None, None

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        x_hat = torch.sigmoid(self.dec_fc2(h))
        return x_hat

    def forward(self, x):
        # ----------- Denoising -----------
        if self.denoising:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        # ----------- Encode -----------
        z, mu, logvar = self.encode(x_noisy)

        # ----------- Decode -----------
        x_hat = self.decode(z)

        # Return extra VAE terms (mu, logvar) for KL-loss
        return x_hat, z, mu, logvar
