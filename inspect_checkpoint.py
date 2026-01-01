import torch
import numpy as np
import sys

# -----------------------------------------------------
# Allow old numpy scalar pickles (your checkpoint uses this)
# -----------------------------------------------------
import torch.serialization
torch.serialization.add_safe_globals([
    np.dtype,
    getattr(np, "_core", np).multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
])


# -----------------------------------------------------
# Load checkpoint safely (weights_only=False)
# -----------------------------------------------------
path = "experiments/hybrid_ann_model_best_byF1.pt"

try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except Exception as e:
    print("\n❌ STILL FAILED. ERROR:")
    print(e)
    sys.exit(0)

print("\n>>> TOP-LEVEL KEYS:", list(ckpt.keys()))

# -----------------------------------------------------
# Try common state-dict keys
# -----------------------------------------------------
for key in ("model_state", "state_dict", "model_state_dict", "state"):
    if isinstance(ckpt, dict) and key in ckpt:
        sd = ckpt[key]
        print(f"\n>>> Using state-dict key: '{key}', entries:", len(sd))
        break
else:
    # maybe ckpt itself is a raw state-dict
    if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
        print("\n>>> Checkpoint is a raw state-dict, entries:", len(sd))
    else:
        print("\n>>> No valid state-dict found. Top-level dump:")
        for i, (k, v) in enumerate(ckpt.items()):
            print(i, k, type(v))
        sys.exit(0)

# -----------------------------------------------------
# Print shapes
# -----------------------------------------------------
print("\n>>> FIRST 80 PARAMS:")
for i, (name, tensor) in enumerate(list(sd.items())[:80]):
    shape = tuple(tensor.shape)
    print(f"{i:03d} {name:40s} {shape}")

print("\n>>> LAST 10 PARAMS:")
for name, tensor in list(sd.items())[-10:]:
    print(f"{name:40s} {tuple(tensor.shape)}")
