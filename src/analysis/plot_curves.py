import os, re, json, glob, argparse
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_trainer_logs(run_dir):
    state = os.path.join(run_dir, 'trainer_state.json')
    if not os.path.exists(state):
        cand = glob.glob(os.path.join(run_dir, '**', 'trainer_state.json'), recursive=True)
        if not cand: return {}
        state = sorted(cand)[-1]
    hist = load_json(state).get('log_history', [])
    data = {
        'train_steps': [],
        'train_loss': [],
        'eval_steps': [],
        'eval_loss': [],
        'eval_f1': [],
        'eval_exact_match': [],
        'eval_rouge_l': []
    }
    for item in hist:
        step = item.get('step')
        if 'loss' in item and step is not None:
            data['train_steps'].append(step)
            data['train_loss'].append(item['loss'])
        if 'eval_loss' in item and step is not None:
            data['eval_steps'].append(step)
            data['eval_loss'].append(item['eval_loss'])
            data['eval_f1'].append(item.get('eval_f1'))
            data['eval_exact_match'].append(item.get('eval_exact_match'))
            data['eval_rouge_l'].append(item.get('eval_rouge_l'))
    return data


def parse_step(path):
    m = re.search(r'gauge_stats_step_(\d+)\.json', os.path.basename(path))
    if m: return int(m.group(1))
    return None


def aggregate_gauge(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, 'analysis', 'gauge_stats_step_*.json')), key=parse_step)
    if not files: return {}
    out = {'steps': [], 'out_scale_mean': [], 'g_attn_norm_mean': [], 'g_val_norm_mean': [], 'g_rel_norm_mean': []}
    for f in files:
        rows = load_json(f)
        out['steps'].append(parse_step(f))
        for key in ['out_scale', 'g_attn_norm', 'g_val_norm', 'g_rel_norm']:
            vals = [r[key] for r in rows if key in r]
            out[f'{key}_mean'].append(sum(vals) / len(vals) if vals else None)
    return out


def safe_plot(xs, ys, label):
    xx = []
    yy = []
    for x, y in zip(xs, ys):
        if y is not None:
            xx.append(x)
            yy.append(y)
    if xx: plt.plot(xx, yy, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    out = args.output_dir or os.path.join(args.run_dir, 'analysis', 'figures')
    ensure_dir(out)
    data = extract_trainer_logs(args.run_dir)
    if data:
        plt.figure(figsize=(8, 5))
        if data['train_steps']:
            plt.plot(data['train_steps'], data['train_loss'], label='train_loss')
        if data['eval_steps']:
            plt.plot(data['eval_steps'], data['eval_loss'], label='eval_loss')
        plt.legend()
        plt.title('Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'loss_curves.png'))
        plt.close()
        plt.figure(figsize=(8, 5))
        safe_plot(data['eval_steps'], data['eval_f1'], 'eval_f1')
        safe_plot(data['eval_steps'], data['eval_exact_match'], 'eval_exact_match')
        safe_plot(data['eval_steps'], data['eval_rouge_l'], 'eval_rouge_l')
        plt.legend()
        plt.title('Eval Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'eval_metrics.png'))
        plt.close()
    g = aggregate_gauge(args.run_dir)
    if g:
        plt.figure(figsize=(8, 5))
        for key in ['out_scale_mean', 'g_attn_norm_mean', 'g_val_norm_mean', 'g_rel_norm_mean']:
            safe_plot(g['steps'], g[key], key)
        plt.legend()
        plt.title('Gauge Couplings')
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'gauge_couplings.png'))
        plt.close()


if __name__ == '__main__': main()
