#!/usr/bin/env python3
"""Analyze classification results and export one subject-level CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

VALID_LABELS = ("chart", "illustration", "Photograph", "Structure")

DEFAULT_INPUT = "./classification/classification_result.json"
DEFAULT_OUT_DIR = "./classification"
DEFAULT_OUTPUT_CSV = "subject_class_6subjects.csv"

SUBJECTS = ("biology", "chemistry", "geography", "materials", "math", "physics")
DATASET_TO_SUBJECT = {
    "EMMA": "chemistry",
    "GEOBench-VLM": "geography",
    "MatCha": "materials",
    "MicroVQA": "biology",
}


def parse_subject(image_key: str) -> str:
    """
    Convert key like:
      image/EMMA/968.png -> physics
      image/MMMU/biology/177.png -> biology
    """
    clean = image_key.strip().replace("\\", "/")
    if clean.startswith("image/"):
        clean = clean[len("image/") :]
    parts = Path(clean).parts
    if not parts:
        return "UNKNOWN"
    if parts[0] == "MMMU" and len(parts) >= 2:
        subject = parts[1].lower()
        return subject if subject in SUBJECTS else "UNKNOWN"
    mapped = DATASET_TO_SUBJECT.get(parts[0], "")
    return mapped if mapped else "UNKNOWN"


def load_results(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got: {type(data)}")
    return {str(k): str(v) for k, v in data.items()}


def analyze(results: Dict[str, str]) -> Tuple[Dict[str, Counter], int]:
    subject_class_counter: Dict[str, Counter] = defaultdict(Counter)
    error_count = 0

    for image_key, label in results.items():
        if label in VALID_LABELS:
            subject = parse_subject(image_key)
            if subject in SUBJECTS:
                subject_class_counter[subject][label] += 1
        else:
            error_count += 1

    for subject in SUBJECTS:
        _ = subject_class_counter[subject]
    return subject_class_counter, error_count


def write_subject_class_csv(path: Path, subject_class_counter: Dict[str, Counter]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", *VALID_LABELS, "total"])
        grand_totals = [0 for _ in VALID_LABELS]
        for subject in SUBJECTS:
            row_counts = [subject_class_counter[subject].get(label, 0) for label in VALID_LABELS]
            for i, cnt in enumerate(row_counts):
                grand_totals[i] += cnt
            writer.writerow([subject, *row_counts, sum(row_counts)])
        writer.writerow(["ALL", *grand_totals, sum(grand_totals)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze classification_result.json and export one CSV for 6 subjects.")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="Path to classification_result.json")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Directory for output CSV files")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_CSV, help="Output CSV file name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    subject_class_counter, error_count = analyze(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_csv = args.out_dir / args.output_name

    write_subject_class_csv(subject_csv, subject_class_counter)

    valid_total = sum(sum(subject_class_counter[s].values()) for s in SUBJECTS)
    print(f"Input: {args.input}")
    print(f"Total entries: {len(results)}")
    print(f"Valid labeled entries in 6 subjects: {valid_total}")
    print(f"Error/invalid entries: {error_count}")
    print(f"Saved: {subject_csv}")


if __name__ == "__main__":
    main()
