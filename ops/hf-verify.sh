#!/usr/bin/env bash
#
# /opt/ops/hf-verify.sh
#
# Verifies that the HuggingFace token is present in the container environment
# and that it authenticates against the Hub. Run from the host:
#
#   docker exec -i rag-benchmark-runner bash /opt/ops/hf-verify.sh
#
# Exit code is non-zero if the token is missing or authentication fails, so it
# is safe to use as a gate before launching benchmarks.

set -euo pipefail

export PYTHONPATH="/opt:${PYTHONPATH:-}"

python - <<'PY'
import os
import sys

ht = os.environ.get("HF_TOKEN", "").strip()
hh = os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()

print("HF_TOKEN set                :", bool(ht))
print("HUGGINGFACE_HUB_TOKEN set   :", bool(hh))

token = ht or hh
if not token:
    print("\nFAIL: no HuggingFace token in the container environment.")
    print("Set HF_TOKEN in .env, then: docker compose ... up -d --force-recreate")
    sys.exit(2)

# Masked echo only — never print the secret.
print("token prefix                :", token[:3] + "..." + token[-2:])

try:
    from huggingface_hub import whoami
    info = whoami(token=token)
    name = info.get("name") if isinstance(info, dict) else str(info)
    print("whoami                      :", name)
except Exception as exc:
    print("\nFAIL: HuggingFace authentication failed:", exc)
    sys.exit(3)

# Confirm access to the previously-gated repo's metadata (no full download).
try:
    from huggingface_hub import model_info
    mi = model_info("google/embeddinggemma-300m", token=token)
    print("gated repo access           : OK  (google/embeddinggemma-300m)")
    print("  sha                       :", getattr(mi, "sha", "n/a"))
except Exception as exc:
    print("\nWARNING: token works but gated repo access failed:", exc)
    print("Make sure you accepted the model licence on the model's HF page.")
    sys.exit(4)

print("\nHF AUTH OK")
PY
