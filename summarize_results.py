#!/usr/bin/env python3
"""Build a Markdown index from CSTrie experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_results import write_results_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 CSTrie results 目录中的实验结果")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    warnings = write_results_summary(args.results_dir, args.output)
    for warning in warnings:
        print(f"[WARN] 跳过无法解析的结果: {warning}")
    print(f"[DONE] {args.output or args.results_dir / 'results_report.md'}")


if __name__ == "__main__":
    main()
