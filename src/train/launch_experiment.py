import argparse
from ..utils.config import load_config
from ..train.train_frozen import run_frozen
from ..train.train_lora import run_lora
from ..train.train_gauge import run_gauge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    method = cfg['experiment']['method'].lower()
    if method == 'frozen': run_frozen(cfg)
    elif method == 'lora': run_lora(cfg)
    elif method == 'gauge': run_gauge(cfg)
    else:
        raise ValueError(method)


if __name__ == '__main__':
    main()
