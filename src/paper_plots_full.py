# src/paper_plots_full.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT = "experiments/results/paper_style"
os.makedirs(OUT, exist_ok=True)

# -----------------------------------------------------------
# Load the combined dataset (with label auto-fix)
# -----------------------------------------------------------
def load_combined():
    npz = np.load("data/combined/combined_dataset.npz", allow_pickle=True)

    adj = npz["adjacency"]                # (8060, 8060)
    n = adj.shape[0]                      # graph nodes = 8060

    features = npz["features"][:n]        # truncate extra rows

    # ------------ FIX: Auto-generate labels if missing ----------
    if "labels" in npz:
        labels = npz["labels"][:n]
        print("✔ Loaded labels from dataset")
    else:
        print("⚠ No labels found — generating synthetic labels from stress features")

        # Use base-paper rule: stress = original + additional
        if features.shape[1] >= 6:
            stress = features[:, 4] + features[:, 5]
        else:
            # Fallback in case features < 6 (should not happen)
            stress = features.sum(axis=1)

        thresh = np.percentile(stress, 90)
        labels = (stress > thresh).astype(int)

        print("➡ Labels generated using stress > 90th percentile")
        print("Label distribution:", np.bincount(labels))

    return adj, features, labels


# -----------------------------------------------------------
# Basic Graphs
# -----------------------------------------------------------
def plot_network_graph(adj, title, out):
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=5, edge_color="gray")
    plt.title(title)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_degree_distribution(adj, out):
    degrees = adj.sum(axis=1)
    plt.figure(figsize=(7, 5))
    plt.hist(degrees, bins=60, edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_correlation(features, out, title="Correlation Matrix"):
    corr = np.corrcoef(features.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="viridis")
    plt.title(title)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_label_graph(adj, labels, out):
    n = adj.shape[0]
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)

    colors = ["red" if x == 1 else "green" for x in labels]

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=colors, node_size=10)
    plt.title("Node Labels Graph")
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------------------------------------
# Stress Scatter
# -----------------------------------------------------------
def plot_stress_scatter(features, out):
    if features.shape[1] < 6:
        print("⚠ Stress scatter requires ≥ 6 features. Skipping.")
        return

    x = features[:, 4]  # original stress
    y = features[:, 5]  # additional stress

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=10, alpha=0.3, c="blue")
    plt.xlabel("Original Stress")
    plt.ylabel("Additional Stress")
    plt.title("Original vs Additional Stress")
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------------------------------------
# PCA + t-SNE
# -----------------------------------------------------------
def plot_pca(features, labels, out, adj=None):
    if adj is not None:
        n = adj.shape[0]
        features = features[:n]
        labels = labels[:n]

    Z2 = PCA(n_components=2).fit_transform(features)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="coolwarm", s=12)
    plt.title("PCA Embeddings")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_tsne(features, labels, out, adj=None):
    if adj is not None:
        n = adj.shape[0]
        features = features[:n]
        labels = labels[:n]

    print("⏳ Computing t-SNE (takes ~45–60 sec)...")
    Z2 = TSNE(n_components=2, perplexity=35, learning_rate=100).fit_transform(features)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title("t-SNE Embeddings")
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------------------------------------
# Train/Test split visualization
# -----------------------------------------------------------
def plot_train_test_graph(adj, labels, out):
    n = len(labels)

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:int(0.8 * n)] = True
    test_mask = ~train_mask

    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)

    node_colors = ["blue" if train_mask[i] else "orange" for i in range(n)]

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=node_colors, node_size=10)
    plt.title("Train/Test Split Visualization")
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def generate_all():
    adj, features, labels = load_combined()

    print("\n📌 Loaded Combined Dataset")
    print("Adjacency:", adj.shape)
    print("Features:", features.shape)
    print("Labels:", labels.shape)

    # A. Network Graphs
    plot_network_graph(adj, "Combined Network Graph", f"{OUT}/network_graph.png")
    plot_degree_distribution(adj, f"{OUT}/degree_distribution.png")
    plot_label_graph(adj, labels, f"{OUT}/label_graph.png")
    plot_train_test_graph(adj, labels, f"{OUT}/train_test_graph.png")

    # B. Dataset Graphs
    plot_correlation(features, f"{OUT}/correlation_matrix.png")
    plot_stress_scatter(features, f"{OUT}/stress_scatter.png")

    # C. Embeddings
    plot_pca(features, labels, f"{OUT}/pca.png", adj)
    plot_tsne(features, labels, f"{OUT}/tsne.png", adj)

    print("\n🔥 ALL PAPER-STYLE GRAPHS GENERATED SUCCESSFULLY")
    print("📁 Saved in:", OUT)


if __name__ == "__main__":
    generate_all()
