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

> Fill in with your own run numbers as you reproduce them. Example progression:

| Stage                                  | Val score |
| -------------------------------------- | --------- |
| DoG blob baseline                      | ~0.78     |
| 3D U-Net detector                      | ~0.85     |
| + patch-bias fix (uniform patches)     | ~0.86     |
| + TTA + ensemble + track filtering     | ~0.87     |

*(Scores depend on the exact competition metric and data split — regenerate locally.)*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the competition data and point the config at it:

```yaml
# config/default.yaml
data:
  root: /path/to/biohub/data
```

Expected layout (adapt in `src/data/dataset.py` to match the actual release):

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

# 3. Predict with TTA + ensemble
python -m src.predict --config config/default.yaml --method unet --tta --ensemble

# 4. Evaluate
python -m src.evaluate --pred outputs/pred.csv --gt data/embryo_05/labels.csv

# 5. Leave-one-embryo-out validation
python -m src.validation.loeo --config config/default.yaml
```

## Notes

- Designed to run on a single modest GPU; falls back to CPU (slow) for the baseline.
- The synthetic-data toggle in `src/data/dataset.py` lets you smoke-test the whole
  pipeline end-to-end without the real dataset.

## License

MIT — see [LICENSE](LICENSE).
