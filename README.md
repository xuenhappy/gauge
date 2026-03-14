# Gauge-QA on DGX Spark (Qwen2.5-32B-Base)

一个面向 DGX Spark 的完整工程模板，用于在 **Qwen2.5-32B-Base** 上验证 **Frozen / LoRA / Gauge** 三种 QA 适配路线。

支持：
- VSCode Remote-SSH + DGX Spark 工作流
- Frozen / LoRA / Gauge 三组统一实验入口
- smoke test 与 SQuAD 小切片实验
- 统一推理入口（自动识别 frozen / lora / gauge）
- Streamlit UI
- 训练后 Gauge 权重的一键加载

## 快速开始

```bash
bash setup.sh
source .venv/bin/activate
cp .env.example .env
source .env
make env-check
```

下载基座模型到 `models/qwen32b_base`：

```bash
make model-download
```

运行：

```bash
make smoke
make make-squad-small
make squad-small
make plots
make samples
make infer RUN_DIR=outputs/runs/gauge_qwen32b_squad_small_v1
make chat RUN_DIR=outputs/runs/gauge_qwen32b_squad_small_v1
make ui
```

## 说明

- 本项目默认以 **Qwen2.5-32B-Base** 为基座。
- Gauge 恢复不能直接 `from_pretrained(final_dir)`；必须先加载 base model，再 patch，再 load checkpoint。
- `src/models/qwen_gauge_attention.py` 是最小可用模板，接入你的本地 Transformers 版本时可能需要按源码小调 forward 签名。
