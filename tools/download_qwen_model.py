#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载 Qwen2.5-32B-Base 到本地目录。

用法示例：

1) 使用默认参数下载
   python tools/download_qwen.py

2) 指定模型与输出目录
   python tools/download_qwen.py \
       --repo_id Qwen/Qwen2.5-32B \
       --output_dir models/qwen32b_base

3) 仅下载特定文件类型
   python tools/download_qwen.py \
       --allow_patterns "*.json" "*.safetensors" "*.model" "*.py" "*.txt"

说明：
- 默认 repo_id 是 Qwen/Qwen2.5-32B（即 base 模型）
- 需要先执行：
    huggingface-cli login
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-32B-Base from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen2.5-32B",
        help="Hugging Face repo id. Default: Qwen/Qwen2.5-32B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/qwen32b_base",
        help="Local output directory. Default: models/qwen32b_base",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision/branch/tag/commit.",
    )
    parser.add_argument(
        "--resume_download",
        action="store_true",
        help="Resume partially completed downloads.",
    )
    parser.add_argument(
        "--local_dir_use_symlinks",
        action="store_true",
        help="Use symlinks in local_dir when supported.",
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional allow patterns, e.g. *.json *.safetensors *.py",
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs="*",
        default=None,
        help="Optional ignore patterns.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved arguments without downloading.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen download tool")
    print(f"repo_id      : {args.repo_id}")
    print(f"output_dir   : {output_dir}")
    print(f"revision     : {args.revision}")
    print(f"cache_dir    : {args.cache_dir}")
    print(f"allow_patterns : {args.allow_patterns}")
    print(f"ignore_patterns: {args.ignore_patterns}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run enabled. Exiting without downloading.")
        return 0

    try:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=args.local_dir_use_symlinks,
            revision=args.revision,
            resume_download=args.resume_download,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            cache_dir=args.cache_dir,
        )
    except HfHubHTTPError as e:
        print("\n[ERROR] Hugging Face Hub request failed.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(
            "\n请确认：\n"
            "1. 已执行 huggingface-cli login\n"
            "2. 网络可访问 Hugging Face\n"
            "3. repo_id 正确\n",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print("\n[ERROR] Download failed.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    print("\nDownload completed.")
    print(f"Local path: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())