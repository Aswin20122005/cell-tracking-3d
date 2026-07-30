#!/usr/bin/env bash
# Train the 3D U-Net, then predict with TTA + ensemble and evaluate.
set -e
python -m src.train   --config config/default.yaml
python -m src.predict --config config/default.yaml --method unet --tta --ensemble --track --out outputs/pred_unet.csv
python -m src.evaluate --pred outputs/pred_unet.csv --gt data/embryo_05/labels.csv
