# SystematicRiskAI

**Hybrid ANN–GNN–Autoencoder Framework for Systemic Risk and Market Anomaly Detection in Financial Networks**

---

## Overview
This repository contains the implementation of a hybrid deep learning framework combining:

- **Artificial Neural Networks (ANN)**
- **Graph Neural Networks (GNN)**
- **Autoencoders (AE)**

to analyze systemic risk and detect market anomalies in financial networks. The framework captures both **node-level features** and **network-level risk propagation**, enabling better prediction of systemic failures in interconnected financial systems.

---

## Project Structure

systematicriskai/
│
├─ src/ # All Python scripts
│ ├─ model/ # ANN, GNN, AE implementations
│ ├─ utils/ # Helper functions, metrics
│ └─ run_experiment.py # Main script to run experiments
│
├─ experiment/ # Notebooks & experiment configs
│ ├─ E1_data_preprocessing.ipynb
│ ├─ E2_training_ANN.ipynb
│ ├─ E3_training_GNN.ipynb
│ ├─ E4_hybrid_model.ipynb
│ └─ E5_results_analysis.ipynb
│
├─ results/ # Generated outputs & figures
│ ├─ metrics/ # CSV/JSON files with evaluation metrics
│ ├─ figures/ # Plots and visualizations
│ ├─ logs/ # Training logs, loss curves
│ └─ summary_report.md # Summary of experiments and insights
│
├─ archive/ # Optional: old checkpoints, unused scripts
│ └─ checkpoint_XYZ.pt
│
├─ data/ # Placeholder for dataset (not included)
│ └─ combined/ # Path for dataset after download
│
├─ .gitignore
├─ README.md
├─ requirements.txt

markdown
Copy code

### Explanation:

- `src/` → all ANN, GNN, AE scripts and helper functions  
- `experiment/` → notebooks showing step-by-step experiments  
- `results/` → figures, metrics, logs to show your research outputs  
- `archive/` → old models, checkpoints, backups  
- `data/` → placeholder for dataset (provide external download link)

> All figures, metrics, and experiment outputs are in the `results/` folder.  
> Large dataset files are excluded; please download from [link].

---

## How to Run

1. Create a virtual environment:

```bash
python -m venv .venv
Activate the environment:

bash
Copy code
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run experiments:

bash
Copy code
python run_experiment.py
⚠️ Note: The full dataset (combined_dataset.npz) is not included due to GitHub size limits.
You can download the dataset from [external link] (Google Drive/Kaggle/etc.) and place it in data/combined/.

Results & Figures
All generated results, metrics, and figures are stored in the results/ folder.

Sample outputs and visualizations are included for reference.

These demonstrate the model’s effectiveness in predicting systemic risk and detecting anomalies.

Citation
If you use this repository for research or publications, please cite the following paper:

rust
Copy code
Hybrid ANN–GNN–Autoencoder Framework for Systemic Risk and Market Anomaly Detection in Financial Networks
Contact
For any questions or collaborations, contact: Harshit Bhatt