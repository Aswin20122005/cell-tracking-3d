#!/usr/bin/env bash
# Whole pipeline on generated data: no dataset or GPU needed (about a minute on CPU).
set -euo pipefail
python -m src.predict --config config/synthetic.yaml --method dog --out outputs/synth_dog.csv
python -m src.train   --config config/synthetic.yaml
python -m src.predict --config config/synthetic.yaml --method unet --tta --track --out outputs/synth_unet.csv
