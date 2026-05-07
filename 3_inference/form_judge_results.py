#!/usr/bin/env python3
"""
Convert generate json files to xlsx files for manual judging.

Input:
  /mnt/shared-storage-user/zhengyuting/SIP/generate/{model_name}/*_generate.json

Output:
  /mnt/shared-storage-user/zhengyuting/SIP/results/{model_name}/{model_name}_{dataset}_{action}.xlsx

Excel columns:
  index, question, answer, prediction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GENERATE_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP/generate")
DEFAULT_RESULTS_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generate json to judge xlsx.")
    parser.add_argument(
        "--generate_root",
        type=Path,
        default=DEFAULT_GENERATE_ROOT,
        help="Root directory of generation results.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory to save xlsx files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root (expect dict): {path}")
    return data


def convert_one_json(json_path: Path, results_root: Path) -> tuple[Path, bool]:
    payload = load_json(json_path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing list field 'results': {json_path}")

    model_name = json_path.parent.name
    dataset = str(payload.get("dataset", "unknown_dataset"))
    action = str(payload.get("action", "unknown_action"))
    run_tag = str(payload.get("run_tag", "") or "").strip()

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "index": item.get("id", idx),
                "question": item.get("query", ""),
                "answer": item.get("target", ""),
                "prediction": item.get("output", ""),
            }
        )

    out_dir = results_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"__{run_tag}" if run_tag else ""
    out_path = out_dir / f"{model_name}_{dataset}_{action}{run_suffix}.xlsx"
    if out_path.exists():
        return out_path, False

    df = pd.DataFrame(rows, columns=["index", "question", "answer", "prediction"])
    df.to_excel(out_path, index=False)
    return out_path, True


def main() -> None:
    args = parse_args()
    generate_root = args.generate_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()

    if not generate_root.exists():
        raise FileNotFoundError(f"Generate root not found: {generate_root}")

    json_files = sorted(generate_root.glob("*/*_generate.json"))
    if not json_files:
        raise FileNotFoundError(f"No generate json found under: {generate_root}")

    print(f"Found {len(json_files)} files.")
    success = 0
    for json_path in json_files:
        try:
            out_path, created = convert_one_json(json_path, results_root)
            if created:
                print(f"[OK] {json_path} -> {out_path}")
            else:
                print(f"[SKIP] {json_path} -> {out_path} already exists.")
            success += 1
        except Exception as e:
            print(f"[ERROR] {json_path}: {e}")

    print(f"Done. success={success}, failed={len(json_files) - success}")


if __name__ == "__main__":
    main()