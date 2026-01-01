import os
import ast
import json
from collections import defaultdict

PROJECT_ROOT = os.getcwd()

REPORT = {
    "folders": [],
    "files": [],
    "python_files": [],
    "data_files": defaultdict(list),
    "file_roles": {},
    "dependencies": defaultdict(list),
    "outputs_detected": defaultdict(list)
}

DATA_EXTENSIONS = (".csv", ".parquet", ".json", ".npy", ".npz", ".xlsx")
OUTPUT_EXTENSIONS = (".png", ".jpg", ".pdf", ".pt", ".pth", ".pkl", ".joblib", ".csv")

ROLE_KEYWORDS = {
    "training": ["train", "fit", "optimizer", "backward"],
    "model": ["nn.Module", "class", "forward"],
    "evaluation": ["roc_auc", "confusion_matrix", "precision", "recall"],
    "plotting": ["matplotlib", "seaborn", "plt.", "plot", "imshow"],
    "data_loader": ["DataLoader", "read_csv", "load", "dataset"]
}

def scan_project():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        REPORT["folders"].append(root)
        for file in files:
            full_path = os.path.join(root, file)
            REPORT["files"].append(full_path)

            if file.endswith(".py"):
                REPORT["python_files"].append(full_path)
                analyze_python_file(full_path)

            if file.endswith(DATA_EXTENSIONS):
                REPORT["data_files"][root].append(file)

            if file.endswith(OUTPUT_EXTENSIONS):
                REPORT["outputs_detected"][root].append(file)

def analyze_python_file(file_path):
    roles = set()
    dependencies = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    dependencies.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module)

        for role, keywords in ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in source:
                    roles.add(role)

        REPORT["file_roles"][file_path] = list(roles)
        REPORT["dependencies"][file_path] = list(dependencies)

    except Exception as e:
        REPORT["file_roles"][file_path] = ["unreadable"]
        REPORT["dependencies"][file_path] = []

def generate_report():
    with open("PROJECT_AUDIT_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=4)

    print("\n====== PROJECT AUDIT SUMMARY ======\n")
    print(f"Total folders      : {len(REPORT['folders'])}")
    print(f"Total files        : {len(REPORT['files'])}")
    print(f"Python files       : {len(REPORT['python_files'])}")
    print(f"Data folders       : {len(REPORT['data_files'])}")
    print(f"Output locations   : {len(REPORT['outputs_detected'])}")

    print("\n--- Python File Roles (High-level) ---")
    for f, roles in REPORT["file_roles"].items():
        print(f"{os.path.basename(f)} -> {roles}")

    print("\nAudit saved to: PROJECT_AUDIT_REPORT.json\n")

if __name__ == "__main__":
    scan_project()
    generate_report()
