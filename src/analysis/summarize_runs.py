import os, json, csv, argparse


def summarize_run(run_dir):
    row = {'run_dir': run_dir, 'config_path': os.path.join(run_dir, 'config.yaml')}
    p = os.path.join(run_dir, 'metrics', 'test_metrics.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            row.update(json.load(f))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dirs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [summarize_run(x) for x in args.run_dirs]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved summary to {args.output}')


if __name__ == '__main__': main()
