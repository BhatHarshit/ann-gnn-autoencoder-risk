import os
import ast
import json
import torch
import platform
import inspect
from pathlib import Path
from datetime import datetime

# ======================================================================
# CONFIG
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
LOG_PATH = PROJECT_ROOT / "experiments/full_system_report.txt"

# ======================================================================
# UTILITIES
# ======================================================================
def header(title):
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}\n")

def log(text):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def both(text):
    print(text)
    log(text)

def reset_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"FULL SYSTEM REPORT — {datetime.now()}\n\n")

# ======================================================================
# 1. PROJECT TREE
# ======================================================================
def print_file_tree():
    header("PROJECT FILE TREE")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(skip in root for skip in ["__pycache__", ".git", "venv", ".venv"]):
            continue
        level = root.replace(str(PROJECT_ROOT), "").count(os.sep)
        indent = " " * (2 * level)
        both(f"{indent}{os.path.basename(root)}/")
        for f in files:
            both(f"{indent}  └── {f}")

# ======================================================================
# 2. PYTHON FILE SUMMARIES
# ======================================================================
def print_python_file_summaries():
    header("PYTHON FILE SUMMARIES")
    for file in SRC_ROOT.rglob("*.py"):
        both(f"\n--- {file} ---")
        try:
            src = file.read_text()
            both(src)
        except Exception as e:
            both(f"Unable to read file: {e}")

# ======================================================================
# 3. STRUCTURE (CLASSES + FUNCTIONS)
# ======================================================================
def analyze_code_structure():
    header("CODE STRUCTURE (CLASSES + FUNCTIONS)")
    for file in SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(file.read_text())
        except Exception:
            continue
        both(f"\n📌 FILE: {file}")
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                both(f"  CLASS: {node.name}")
            elif isinstance(node, ast.FunctionDef):
                both(f"  FUNCTION: {node.name}")

# ======================================================================
# 4. IMPORT RELATIONSHIPS
# ======================================================================
def inspect_imports():
    header("IMPORT RELATIONSHIPS")
    for file in SRC_ROOT.rglob("*.py"):
        both(f"\n📁 {file}")
        try:
            tree = ast.parse(file.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    both(f"  import {n.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for n in node.names:
                    both(f"  from {module} import {n.name}")

# ======================================================================
# 5. DEPENDENCY GRAPH
# ======================================================================
def parse_imports(file_path):
    try:
        tree = ast.parse(Path(file_path).read_text())
    except Exception:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module if node.module else "")
    return imports

def dependency_graph():
    header("DEPENDENCY GRAPH")
    py_files = list(PROJECT_ROOT.rglob("*.py"))
    for file in py_files:
        imports = parse_imports(file)
        both(f"\n📄 {file}")
        for imp in imports:
            both(f"   → imports: {imp}")

# ======================================================================
# 6. CHECKPOINT STRUCTURE
# ======================================================================
def inspect_checkpoint():
    header("CHECKPOINT STRUCTURE")
    ckpt_path = PROJECT_ROOT / "experiments/hybrid_ann_model_best_byF1.pt"

    if not ckpt_path.exists():
        both("❌ No checkpoint found.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    both(f"Top-level keys: {list(ckpt.keys())}")

    sd = None
    for key in ["model_state", "state_dict", "model_state_dict", "state"]:
        if key in ckpt:
            sd = ckpt[key]
            both(f"✔ Using state-dict key: {key} ({len(sd)} params)")
            break

    if sd is None:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            both("❌ No valid state-dict found.")
            return

    both("\nFIRST 50 params:")
    for i, (k, v) in enumerate(list(sd.items())[:50]):
        both(f"{i:03d} {k:40s} {tuple(v.shape)}")

    both("\nLAST 20 params:")
    for i, (k, v) in enumerate(list(sd.items())[-20:]):
        both(f"{i:03d} {k:40s} {tuple(v.shape)}")

# ======================================================================
# 7. MODEL FORWARD TEST
# ======================================================================
def test_model_forward():
    header("FORWARD PASS TEST")
    try:
        from src.model_hybrid_ann import HybridGNN_AE_ANN
    except Exception as e:
        both(f"❌ Failed to import model: {e}")
        return

    model = HybridGNN_AE_ANN()
    both(f"Model loaded: {model.__class__.__name__}")

    X = torch.randn(10, 6)
    A = torch.eye(10)

    try:
        X_recon, Z, logits, X_ann = model(X, A)
        both("✔ Forward pass SUCCESS")
        both(f"  X_recon: {X_recon.shape}")
        both(f"  Z:       {Z.shape}")
        both(f"  logits:  {logits.shape}")
        both(f"  X_ann:   {X_ann.shape}")
    except Exception as e:
        both(f"❌ Forward pass failed: {e}")

# ======================================================================
# 8. ENVIRONMENT + PYTORCH INFO
# ======================================================================
def environment_report():
    header("ENVIRONMENT REPORT")
    both(f"Python: {platform.python_version()}")
    both(f"Platform: {platform.platform()}")
    both(f"Torch version: {torch.__version__}")
    both(f"CUDA available: {torch.cuda.is_available()}")
    both(f"CUDA device count: {torch.cuda.device_count()}")

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    reset_log()

    print_file_tree()
    print_python_file_summaries()
    analyze_code_structure()
    inspect_imports()
    dependency_graph()
    inspect_checkpoint()
    test_model_forward()
    environment_report()

    header("REPORT COMPLETE")
    both(f"Report saved to: {LOG_PATH}")
