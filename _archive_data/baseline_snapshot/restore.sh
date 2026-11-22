#!/bin/bash
# Roll production back to the models of this snapshot.
set -e
cd "$(dirname "$0")/../.."
mkdir -p models/lgbm
[ -d models/lgbm/20261109-204711 ] || cp -r models/baseline_snapshot/production/lgbm/20261109-204711 models/lgbm/
echo -n 20261109-204711 > models/lgbm/latest.txt
mkdir -p models/lstm
[ -d models/lstm/20261109-204722 ] || cp -r models/baseline_snapshot/production/lstm/20261109-204722 models/lstm/
echo -n 20261109-204722 > models/lstm/latest.txt
mkdir -p models/tft
[ -d models/tft/20261109-204738 ] || cp -r models/baseline_snapshot/production/tft/20261109-204738 models/tft/
echo -n 20261109-204738 > models/tft/latest.txt
cp models/baseline_snapshot/processed/model_selection.json data/processed/model_selection.json
cp models/baseline_snapshot/processed/postprocess.json data/processed/postprocess.json
cp models/baseline_snapshot/processed/best_model.txt data/processed/best_model.txt
echo "Production restored to the baseline snapshot. Run: python scripts/predict.py"
