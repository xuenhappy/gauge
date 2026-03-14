SHELL := /bin/bash
PY := python
VENV_ACTIVATE := source .venv/bin/activate
ENV_ACTIVATE := source .env
RUN_DIR ?= outputs/runs/gauge_qwen32b_squad_small_v1

install:
	bash setup.sh

env-check:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && $(PY) tools/check_dgx_spark.py

model-download:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && $(PY) tools/download_qwen_model.py

make-squad-small:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && $(PY) tools/make_squad_small.py --output_dir data/squad --train_size 2000 --dev_size 200 --test_size 200

smoke:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/run_smoke_all.sh

squad-small:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/run_squad_small_all.sh

plots:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/plot_all.sh

samples:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/generate_all_samples.sh

ui:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/run_streamlit.sh

chat:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/chat.sh $(RUN_DIR)

infer:
	$(VENV_ACTIVATE) && $(ENV_ACTIVATE) && bash scripts/test_infer.sh $(RUN_DIR)
