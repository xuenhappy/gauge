import os, yaml
from ..eval.evaluate import run_evaluation


def run_frozen(cfg):
    os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)
    with open(os.path.join(cfg['experiment']['output_dir'], 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return run_evaluation(cfg, cfg['model']['base_model_name_or_path'], cfg['experiment']['output_dir'])
