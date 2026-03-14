#!/usr/bin/env bash
set -e
python -m src.eval.generate_samples --sample_file data/smoke/sample_cases.jsonl --prompt_style qa_standard --output_dir outputs/figures/sample_comparisons --run frozen=outputs/runs/frozen_qwen32b_squad_small_v1 --run lora=outputs/runs/lora_qwen32b_squad_small_v1 --run gauge=outputs/runs/gauge_qwen32b_squad_small_v1
