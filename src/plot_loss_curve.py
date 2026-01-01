"""
plot_loss_curve.py
Visualizes the training loss curve stored in log file or generated manually.
"""

import matplotlib.pyplot as plt

# Manually enter your observed losses (from train_hybrid.py output)
epochs = [1, 10, 20, 30, 40, 50]
losses = [0.2995, 0.2349, 0.1723, 0.1227, 0.0937, 0.0897]  # replace with yours

plt.figure(figsize=(6,4))
plt.plot(epochs, losses, marker='o', linewidth=2)
plt.title("Training Loss Curve (Hybrid GNN + Autoencoder)")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("experiments/results/loss_curve.png")
plt.show()
print("📊 Saved loss curve at experiments/results/loss_curve.png")
