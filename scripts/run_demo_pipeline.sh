#!/usr/bin/env bash
set -e
source .venv/bin/activate
source .env
python tools/check_dgx_spark.py
python tools/make_squad_small.py --output_dir data/squad --train_size 2000 --dev_size 200 --test_size 200
bash scripts/run_smoke_all.sh
bash scripts/run_squad_small_all.sh
bash scripts/plot_all.sh
bash scripts/generate_all_samples.sh
