#!/usr/bin/env python3
"""
Convert original QA json for generation usage.

Usage:
- Input: any QA json file path via `--input`
- Output: target json file path via `--output`
- Replace every item in `image_path`: `image/...` -> `original_image/...`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASET_TO_FIELD = {
    "EMMA": "chemistry",
    "GEOBench-VLM": "geography",
    "MatCha": "materials",
    "MicroVQA": "biology",
}


def replace_image_prefix(path: str) -> str:
    if path.startswith("image/"):
        return path.replace("image/", "original_image/", 1)
    return path


def infer_field(input_path: Path) -> str:
    dataset = input_path.parent.name
    subject = input_path.stem.replace("_qa", "")
    return DATASET_TO_FIELD.get(dataset, subject)


def convert_records(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    for item in records:
        image_paths = item.get("image_path")
        if isinstance(image_paths, list):
            item["image_path"] = [replace_image_prefix(str(p)) for p in image_paths]
        if "field" not in item:
            item["field"] = field
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert original QA json image_path prefix.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input qa json file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output json file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list.")

    field = infer_field(args.input)
    converted = convert_records(data, field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} records -> {args.output}")


if __name__ == "__main__":
    main()