#!/usr/bin/env bash
set -e
for RUN in outputs/runs/frozen_qwen32b_smoke_v1 outputs/runs/lora_qwen32b_smoke_v1 outputs/runs/gauge_qwen32b_smoke_v1 outputs/runs/frozen_qwen32b_squad_small_v1 outputs/runs/lora_qwen32b_squad_small_v1 outputs/runs/gauge_qwen32b_squad_small_v1; do
  if [ -d "$RUN" ]; then
    python -m src.analysis.plot_curves --run_dir "$RUN"
  fi
done
