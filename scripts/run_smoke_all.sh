#!/usr/bin/env bash
set -e
python -m src.train.launch_experiment --config configs/base/frozen_smoke.yaml
python -m src.train.launch_experiment --config configs/base/lora_smoke.yaml
python -m src.train.launch_experiment --config configs/base/gauge_smoke.yaml
python -m src.analysis.summarize_runs --run_dirs outputs/runs/frozen_qwen32b_smoke_v1 outputs/runs/lora_qwen32b_smoke_v1 outputs/runs/gauge_qwen32b_smoke_v1 --output outputs/tables/smoke_summary.csv
