import os, yaml

def load_run_config(run_dir: str):
    p = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(p): raise FileNotFoundError(p)
    with open(p, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def detect_run_method(run_dir: str) -> str:
    method = load_run_config(run_dir).get('experiment', {}).get('method')
    if method is None: raise ValueError('Missing experiment.method')
    method = method.lower()
    if method not in {'frozen','lora','gauge'}: raise ValueError(method)
    return method

def get_final_dir(run_dir: str) -> str:
    p = os.path.join(run_dir, 'final')
    if not os.path.exists(p): raise FileNotFoundError(p)
    return p
