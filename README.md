# 3D + Time Cell Tracking (Kaggle Biohub)

End-to-end data and ML pipeline for **detecting and tracking cells across 3D microscopy
volumes over time** — built for the Biohub zebrafish cell-tracking task. The pipeline
progresses from a classical blob-detection baseline to a trained **3D U-Net heatmap
detector**, with a leave-one-embryo-out validation harness, patch-based training,
test-time augmentation, and lightweight ensembling.

---

## What's inside

```
cell-tracking-3d/
├── config/                 # YAML run configs
├── src/
│   ├── data/               # volume loading, patch sampling, augmentation
│   ├── models/             # 3D U-Net heatmap detector
│   ├── detection/          # DoG blob baseline + heatmap peak extraction
│   ├── tracking/           # frame-to-frame linking across time
│   ├── validation/         # leave-one-embryo-out (LOEO) harness
│   ├── inference/          # test-time augmentation + ensembling
│   ├── train.py            # training entrypoint
│   ├── predict.py          # inference entrypoint
│   └── evaluate.py         # scoring against ground truth
├── notebooks/              # data exploration
├── scripts/                # convenience shell wrappers
└── tests/                  # sanity tests
```

## Approach

1. **Baseline — Difference of Gaussians (DoG).** A classical blob detector over each
   3D volume gives a fast, no-training reference and a sanity check on the data.
2. **3D U-Net heatmap detector.** The network predicts a Gaussian heatmap centred on
   each cell. Local peaks in the predicted heatmap (with non-max suppression) become
   detections. Trained on 3D patches under a strict compute budget.
3. **Patch sampling.** Early versions sampled patches centred on ground-truth cells,
   which biased the model toward always-firing. Adding **uniform random patches**
   alongside GT-centred ones fixed the class balance and cut false positives.
4. **Leave-one-embryo-out (LOEO) validation.** Because embryos differ, a naive random
   split leaks information across frames of the same embryo. LOEO holds out a whole
   embryo per fold and gives a trustworthy estimate — it's what surfaced the sampling
   bug above.
5. **Test-time augmentation + ensembling.** Averaging heatmaps over flips/rotations and
   over a small seed ensemble stabilises detections and trims spurious peaks.
6. **Tracking.** Detections are linked frame-to-frame with a nearest-neighbour /
   Hungarian assignment, with short-track filtering to drop transient false positives.

## Results

**Synthetic data** (reproducible on any machine, `scripts/run_synthetic.sh`, held-out embryo, matching tolerance 3 voxels):

| Method | Precision | Recall | F1 |
| --- | --- | --- | --- |
| DoG blob baseline | 1.00 | 0.99 | 0.99 |
| 3D U-Net, 3 epochs | 0.16 | 1.00 | 0.27 |
| 3D U-Net, 12 epochs | 1.00 | 1.00 | 1.00 |

The synthetic cells are easy (isotropic Gaussians plus mild noise), so these numbers check that the pipeline works, not how well the method performs. The 3-epoch row shows the same "always firing" behaviour an under-trained detector has on real data.

**Kaggle Biohub data:** the competition runs progressed from a DoG baseline to the 3D U-Net with TTA, ensembling and track filtering (leaderboard ~0.78 → ~0.87). Scores depend on the exact competition metric and data split; regenerate locally with the commands below.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# CPU-only machines: install the much smaller CPU build of PyTorch first
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Quick start without the dataset

```bash
scripts/run_synthetic.sh      # DoG baseline, train a small U-Net, predict with TTA + tracking (~1 min on CPU)
```

## Data

Download the competition data and point the config at it:

```yaml
# config/default.yaml
data:
  root: /path/to/biohub/data
```

Expected layout (loaded by `src/data/dataset.py`; adapt it if the release differs):

```
data/
├── embryo_01/
│   ├── volumes/     # t0000.tif, t0001.tif, ...   (Z, Y, X per frame)
│   └── labels.csv   # t, z, y, x  (one row per cell)
├── embryo_02/
└── ...
```

## Usage

```bash
# 1. Classical baseline (no training)
python -m src.predict --config config/default.yaml --method dog

# 2. Train the 3D U-Net
python -m src.train --config config/default.yaml

# 3. Predict with TTA + ensemble (+ tracking)
scripts/train_ensemble.sh            # trains seeds 0, 1, 2
python -m src.predict --config config/default.yaml --method unet --tta --ensemble --track

# 4. Evaluate (--embryo selects the embryo the labels belong to)
python -m src.evaluate --pred outputs/pred.csv --gt data/embryo_05/labels.csv --embryo embryo_05

# 5. Leave-one-embryo-out validation
python -m src.validation.loeo --config config/default.yaml
```

## Configuration notes

- `data.voxel_size` (z, y, x) is used for linking and matching distances, so anisotropic stacks (thick z-slices) are handled in physical units.
- `track.max_gap` is how many frames a cell may go undetected before its track is closed.
- Config files are validated on load; missing keys are reported together.

## Tests

```bash
pip install -r requirements-dev.txt
pytest          # 46 tests, CPU only, ~15 s
```

The suite covers data loading (synthetic and the real TIFF layout), patch sampling, augmentation and TTA inverses, the U-Net, sliding-window inference, ensembling, peak extraction, metrics, tracking, config validation, and end-to-end training → prediction → LOEO on synthetic data. GitHub Actions runs it on every push with CPU-only PyTorch.

## Notes

- Designed to run on a single modest GPU; falls back to CPU (slow on real data).
- `src/data/` (dataset, patches, transforms) was missing from earlier versions of this repository, because the `.gitignore` rule `data/` also matched `src/data/`. The rules are now anchored to the repository root.

## License

MIT — see [LICENSE](LICENSE).
