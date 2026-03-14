#!/usr/bin/env bash
set -e
source .venv/bin/activate
source .env
streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
