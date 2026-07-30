#!/usr/bin/env bash
# Classical DoG baseline over the validation embryo.
set -e
python -m src.predict --config config/default.yaml --method dog --out outputs/pred_dog.csv
python -m src.evaluate --pred outputs/pred_dog.csv --gt data/embryo_05/labels.csv
