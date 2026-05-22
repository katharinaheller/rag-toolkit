"""Copy-paste cells for a JupyterLab notebook to verify HF auth at runtime.

Run inside the GPU notebook container (admin) or the benchmark-runner.
Each block below is one notebook cell.
"""

# ── Cell 1: token presence ───────────────────────────────────────────────
import os
print("HF_TOKEN present              :", bool(os.environ.get("HF_TOKEN")))
print("HUGGINGFACE_HUB_TOKEN present :", bool(os.environ.get("HUGGINGFACE_HUB_TOKEN")))


# ── Cell 2: who am I (authenticated identity) ─────────────────────────────
from huggingface_hub import whoami
print(whoami())


# ── Cell 3: load the previously-gated embedding model ─────────────────────
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
vec = model.encode(["gated model now loads with the forwarded token"])
print("embedding shape:", vec.shape)
