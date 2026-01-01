import json
from collections import Counter

with open("PROJECT_AUDIT_REPORT.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n===== PROJECT AUDIT SUMMARY (CONDENSED) =====\n")

print("📁 Top-level folders:")
folders = set(p.split("/")[0] for p in data["folders"])
for f in sorted(folders):
    print(" -", f)

print("\n🐍 Python file role distribution:")
role_counter = Counter()
for roles in data["file_roles"].values():
    for r in roles:
        role_counter[r] += 1
for r, c in role_counter.items():
    print(f" {r}: {c}")

print("\n📊 Files likely generating GRAPHS:")
for f, roles in data["file_roles"].items():
    if "plotting" in roles:
        print(" -", f)

print("\n🧠 Files likely TRAINING models:")
for f, roles in data["file_roles"].items():
    if "training" in roles:
        print(" -", f)

print("\n📈 Files likely doing EVALUATION:")
for f, roles in data["file_roles"].items():
    if "evaluation" in roles:
        print(" -", f)

print("\n💾 Output files detected (plots/models):")
for folder, files in data["outputs_detected"].items():
    if files:
        print(folder)
        for f in files:
            print("   ", f)
