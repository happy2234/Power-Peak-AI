#!/bin/bash
# Roll production back to the models of this snapshot.
set -e
cd "$(dirname "$0")/../.."
mkdir -p models/lgbm
[ -d models/lgbm/20261109-204711 ] || cp -r models/pre_weather_snapshot/production/lgbm/20261109-204711 models/lgbm/
echo -n 20261109-204711 > models/lgbm/latest.txt
mkdir -p models/lstm
[ -d models/lstm/20261109-204722 ] || cp -r models/pre_weather_snapshot/production/lstm/20261109-204722 models/lstm/
echo -n 20261109-204722 > models/lstm/latest.txt
mkdir -p models/tft
[ -d models/tft/20261109-204738 ] || cp -r models/pre_weather_snapshot/production/tft/20261109-204738 models/tft/
echo -n 20261109-204738 > models/tft/latest.txt
cp models/pre_weather_snapshot/processed/model_selection.json data/processed/model_selection.json
cp models/pre_weather_snapshot/processed/postprocess.json data/processed/postprocess.json
cp models/pre_weather_snapshot/processed/best_model.txt data/processed/best_model.txt
cp models/pre_weather_snapshot/processed/features.parquet data/processed/features.parquet
cp models/pre_weather_snapshot/processed/weather_fc.parquet data/processed/weather_fc.parquet
cp models/pre_weather_snapshot/processed/region_weights.json data/processed/region_weights.json
cp models/pre_weather_snapshot/processed/weather_normals.json data/processed/weather_normals.json
cp models/pre_weather_snapshot/processed/weather_city.parquet data/processed/weather_city.parquet
cp models/pre_weather_snapshot/processed/latest_weather.parquet data/processed/latest_weather.parquet
cp models/pre_weather_snapshot/processed/latest_raw_weather.parquet data/processed/latest_raw_weather.parquet
cp models/pre_weather_snapshot/processed/latest_explanations.parquet data/processed/latest_explanations.parquet
cp models/pre_weather_snapshot/processed/weather_regimes.parquet data/processed/weather_regimes.parquet
cp models/pre_weather_snapshot/processed/backtest_predictions.parquet data/processed/backtest_predictions.parquet
echo "Production restored to the snapshot. Run: python scripts/predict.py"
