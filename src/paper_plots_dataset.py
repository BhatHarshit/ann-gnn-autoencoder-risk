# src/paper_plots_dataset.py
"""
FULL BASE-PAPER STYLE DATASET GRAPHS
(Independent from model. Uses only dataset.)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

OUT = "experiments/results/paper_style"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
def load_dataset():
    npz = np.load("data/combined/combined_dataset.npz", allow_pickle=True)

    # Your correct keys:
    adj = npz["adjacency"]
    features = npz["features"]

    # If labels missing → auto-generate from stress rule
    if "labels" in npz:
        labels = npz["labels"]
        print("✔ Loaded labels from dataset")
    else:
        print("⚠ No labels found — auto-generating (stress > 90th percentile)")

        if features.shape[1] >= 6:
            stress = features[:, 4] + features[:, 5]
        else:
            stress = features.sum(axis=1)

        thresh = np.percentile(stress, 90)
        labels = (stress > thresh).astype(int)

        print("Label distribution:", np.bincount(labels))

    return adj, features, labels


# ---------------------------------------------------
# BASIC STRUCTURAL GRAPHS
# ---------------------------------------------------
def adjacency_heatmap(adj):
    plt.figure(figsize=(6, 5))
    plt.imshow(adj, cmap="inferno")
    plt.title("Adjacency Matrix Heatmap")
    plt.colorbar()
    plt.savefig(f"{OUT}/adjacency_heatmap.png")
    plt.close()


def degree_distribution(adj):
    degrees = adj.sum(axis=1)
    plt.figure(figsize=(6, 5))
    plt.hist(degrees, bins=60, edgecolor='black')
    plt.title("Degree Distribution")
    plt.savefig(f"{OUT}/degree_distribution.png")
    plt.close()


def shortest_path_dist(G):
    spl = dict(nx.all_pairs_shortest_path_length(G))
    lengths = []
    for src in spl:
        for dst in spl[src]:
            if src != dst:
                lengths.append(spl[src][dst])

    plt.figure(figsize=(6, 5))
    plt.hist(lengths, bins=40, edgecolor='black')
    plt.title("Shortest Path Length Distribution")
    plt.savefig(f"{OUT}/shortest_path_lengths.png")
    plt.close()


def eigenvalue_spectrum(adj):
    vals = np.linalg.eigvalsh(adj)
    plt.figure(figsize=(6, 4))
    plt.plot(sorted(vals))
    plt.title("Eigenvalue Spectrum")
    plt.savefig(f"{OUT}/eigenvalue_spectrum.png")
    plt.close()


def clustering_coeff(G):
    c = list(nx.clustering(G).values())
    plt.figure(figsize=(6, 5))
    plt.hist(c, bins=50, edgecolor='black')
    plt.title("Clustering Coefficient Distribution")
    plt.savefig(f"{OUT}/clustering_coeff.png")
    plt.close()


def connected_components(G):
    comps = [len(c) for c in nx.connected_components(G)]
    plt.figure(figsize=(6, 5))
    plt.hist(comps, bins=40, edgecolor="black")
    plt.title("Component Size Distribution")
    plt.savefig(f"{OUT}/component_sizes.png")
    plt.close()


# ---------------------------------------------------
# CENTRALITY PLOTS
# ---------------------------------------------------
def centrality_plots(G):
    deg_cent = list(nx.degree_centrality(G).values())
    bet_cent = list(nx.betweenness_centrality(G).values())
    clo_cent = list(nx.closeness_centrality(G).values())

    plt.figure()
    plt.hist(deg_cent, bins=40)
    plt.title("Degree Centrality")
    plt.savefig(f"{OUT}/degree_centrality.png")
    plt.close()

    plt.figure()
    plt.hist(bet_cent, bins=40)
    plt.title("Betweenness Centrality")
    plt.savefig(f"{OUT}/betweenness_centrality.png")
    plt.close()

    plt.figure()
    plt.hist(clo_cent, bins=40)
    plt.title("Closeness Centrality")
    plt.savefig(f"{OUT}/closeness_centrality.png")
    plt.close()


# ---------------------------------------------------
# FEATURE CORRELATION
# ---------------------------------------------------
def feature_corr(features):
    corr = np.corrcoef(features.T)
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.savefig(f"{OUT}/feature_correlation.png")
    plt.close()


# ---------------------------------------------------
# GRAPH LAYOUT VISUAL
# ---------------------------------------------------
def graph_layout(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=8, edge_color="gray")
    plt.title("Graph Layout (Spring)")
    plt.savefig(f"{OUT}/graph_layout.png")
    plt.close()


# ---------------------------------------------------
# PCA OF FEATURES
# ---------------------------------------------------
def pca_features(features, labels):
    Z2 = PCA(n_components=2).fit_transform(features)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title("PCA of Node Features")
    plt.savefig(f"{OUT}/pca_features.png")
    plt.close()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def generate_all():
    print("📌 Loading dataset...")
    adj, features, labels = load_dataset()

    print("📌 Constructing graph...")
    G = nx.from_numpy_array(adj)

    print("📸 Generating dataset-level graphs...")

    adjacency_heatmap(adj)
    degree_distribution(adj)
    eigenvalue_spectrum(adj)
    clustering_coeff(G)
    connected_components(G)
    centrality_plots(G)
    feature_corr(features)
    graph_layout(G)
    pca_features(features, labels)

    print("\n✅ Dataset Graphs Completed Successfully.")
    print("📁 Saved to:", OUT)


if __name__ == "__main__":
    generate_all()
