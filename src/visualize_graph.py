"""
visualize_graph.py
Plots the financial network (SymNet) showing node connections and anomalies.
"""

import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from data_loader import generate_symnet

# Load generated data
adj, features = generate_symnet(num_nodes=30)

# Build graph
G = nx.from_numpy_array(adj)

# Load anomaly scores if available
score_path = "experiments/results/anomaly_scores.csv"
anomalies = None
if os.path.exists(score_path):
    scores = np.loadtxt(score_path, delimiter=",")
    threshold = np.mean(scores) + 2 * np.std(scores)
    anomalies = np.where(scores > threshold)[0]
else:
    scores = np.zeros(adj.shape[0])

# Node color map (red = anomaly)
colors = ["red" if i in anomalies else "skyblue" for i in range(len(G.nodes))]

plt.figure(figsize=(6,6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, edge_color="gray")
plt.title("Financial Network (SymNet) with Anomalous Nodes Highlighted")
plt.tight_layout()

os.makedirs("experiments/results", exist_ok=True)
plt.savefig("experiments/results/network_visual.png")
plt.show()
print("📊 Saved visualization: experiments/results/network_visual.png")
