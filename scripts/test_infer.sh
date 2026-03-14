#!/usr/bin/env bash
set -e
RUN_DIR=$1
if [ -z "$RUN_DIR" ]; then echo "Usage: bash scripts/test_infer.sh <run_dir>"; exit 1; fi
source .venv/bin/activate
source .env
python -m src.eval.chat --run_dir "$RUN_DIR" --context "Paris is the capital of France." --question "What is the capital of France?"
