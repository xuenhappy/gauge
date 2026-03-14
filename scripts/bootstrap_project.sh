#!/usr/bin/env bash
set -e
bash setup.sh
source .venv/bin/activate
source .env
python tools/check_dgx_spark.py
echo "Bootstrap complete."
