# src/generate_comparison_table.py
import os
import numpy as np
import csv

def read_metrics(path="experiments/results/metrics_summary.txt"):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path, "r") as f:
        for line in f:
            k, v = line.strip().split(",")
            metrics[k] = float(v)
    return metrics

def generate_table(base_metrics=None, out_csv="experiments/results/comparison_table.csv"):
    # base_metrics: dict with baseline numbers from base paper (fill these)
    if base_metrics is None:
        base_metrics = {
            "Mean_recon_err": 0.077,
            "Std_recon_err": 0.045,
            "ROC_AUC": 0.91,
            "Precision": 0.80,
            "Recall": 0.75,
            "F1": 0.77
        }
    current = read_metrics()
    # compose CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = [
        ["Metric", "Base paper (2023)", "Your implementation (current)"]
    ]
    for k, base_val in base_metrics.items():
        cur_val = current.get(k, "NA")
        rows.append([k, base_val, cur_val])
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("✅ Comparison table written to", out_csv)

if __name__ == "__main__":
    generate_table()
