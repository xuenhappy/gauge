#!/usr/bin/env bash
set -e
mkdir -p configs/base data/smoke data/squad data/hotpot models outputs/runs outputs/tables outputs/figures outputs/logs docs src/{models,data,train,eval,analysis,ui,utils} scripts tools
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
echo "Setup complete."
