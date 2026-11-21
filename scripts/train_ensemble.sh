#!/usr/bin/env bash
# Train a 3-seed ensemble and predict with TTA + ensembling.
# Usage: scripts/train_ensemble.sh [config]   (default config/default.yaml)
set -euo pipefail
CONFIG="${1:-config/default.yaml}"
for seed in 0 1 2; do
  python -m src.train --config "$CONFIG" --seed "$seed"
done
python - "$CONFIG" <<'PY'
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path))
ckpt_dir = cfg["train"]["checkpoint_dir"]
print("Add to", path, "under inference.ensemble_checkpoints:")
for s in (0, 1, 2):
    print(f"  - {ckpt_dir}/detector_s{s}.pt")
PY
