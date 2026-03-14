import os
import json
import random
import argparse
from datasets import load_dataset


def convert_example(ex, idx):
    answers = ex.get("answers", {})
    answer_text = ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts:
            answer_text = texts[0]
    elif isinstance(answers, list) and len(answers) > 0:
        answer_text = answers[0]

    return {
        "id": ex.get("id", str(idx)),
        "context": ex["context"],
        "question": ex["question"],
        "answer": answer_text,
    }


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/squad")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--dev_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    ds = load_dataset("squad")

    train = list(ds["train"])
    val = list(ds["validation"])

    random.shuffle(train)
    random.shuffle(val)

    train_small = [convert_example(ex, i) for i, ex in enumerate(train[:args.train_size])]
    dev_small = [convert_example(ex, i) for i, ex in enumerate(val[:args.dev_size])]
    test_small = [convert_example(ex, i) for i, ex in enumerate(val[args.dev_size:args.dev_size + args.test_size])]

    save_jsonl(train_small, os.path.join(args.output_dir, "train_small.jsonl"))
    save_jsonl(dev_small, os.path.join(args.output_dir, "dev_small.jsonl"))
    save_jsonl(test_small, os.path.join(args.output_dir, "test_small.jsonl"))

    print(f"Saved train_small={len(train_small)}")
    print(f"Saved dev_small={len(dev_small)}")
    print(f"Saved test_small={len(test_small)}")


if __name__ == "__main__":
    main()