#!/usr/bin/env python3
"""
Convert QA json files under qa_for_generate to TSV custom data files.

Rules:
- Input root:  /mnt/shared-storage-user/zhengyuting/SIP/qa_for_generate
- Output root: /mnt/shared-storage-user/zhengyuting/SIP/custom_data
- Output file name keeps original file name (no subdirectories), extension .tsv
- Columns: index, question, answer, image
- image column stores base64-encoded image data (not file path)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_QA_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP/qa_for_generate")
DEFAULT_OUTPUT_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP/custom_data")
DEFAULT_SIP_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert qa_for_generate json to custom TSV.")
    parser.add_argument("--qa_root", type=Path, default=DEFAULT_QA_ROOT, help="Input qa root.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output tsv root.")
    parser.add_argument("--image_root", type=Path, default=DEFAULT_SIP_ROOT, help="SIP root for relative image paths.")
    parser.add_argument(
        "--run_tags",
        type=str,
        default="",
        help="Comma-separated run tags to duplicate outputs, e.g. run01,run02,run03. "
        "Use empty string to disable duplication.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(raw_path: str, image_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return image_root / p


def encode_images(image_paths: list[Any], image_root: Path) -> str:
    encoded_list: list[str] = []
    for p in image_paths:
        abs_path = resolve_image_path(str(p), image_root)
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        encoded_list.append(encode_image_base64(abs_path))

    if len(encoded_list) == 1:
        return encoded_list[0]
    # Multiple images: keep all data in JSON-string form.
    return json.dumps(encoded_list, ensure_ascii=False)


def convert_one_json(input_json: Path, output_tsv: Path, image_root: Path) -> tuple[int, int]:
    data = load_json(input_json)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON root must be a list: {input_json}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            skipped += 1
            continue

        question = item.get("query")
        answer = item.get("target")
        image_paths = item.get("image_path")

        if not isinstance(question, str) or answer is None or not isinstance(image_paths, list) or not image_paths:
            skipped += 1
            continue

        encoded_image = encode_images(image_paths, image_root)
        rows.append(
            {
                "index": item.get("id", idx),
                "question": question,
                "answer": str(answer),
                "image": encoded_image,
            }
        )

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "question", "answer", "image"],
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), skipped


def main() -> None:
    args = parse_args()
    qa_root = args.qa_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    run_tags = [x.strip() for x in str(args.run_tags).split(",") if x.strip()]

    if not qa_root.exists():
        raise FileNotFoundError(f"qa_root not found: {qa_root}")

    json_files = sorted(qa_root.glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No json files under: {qa_root}")

    print(f"Found {len(json_files)} json files.")
    done = 0
    for json_file in json_files:
        try:
            base_output_tsv = output_root / f"{json_file.stem}.tsv"
            if run_tags:
                tagged_outputs = [output_root / f"{json_file.stem}__{tag}.tsv" for tag in run_tags]
                existing_tagged = [p for p in tagged_outputs if p.exists()]
                missing_tagged = [p for p in tagged_outputs if not p.exists()]

                if not missing_tagged:
                    print(f"[SKIP] {json_file.name} -> all targets exist, skip.")
                    done += 1
                    continue

                written, skipped = convert_one_json(json_file, base_output_tsv, image_root)
                generated_names = []
                skipped_names = [p.name for p in existing_tagged]
                for tagged_output_tsv in missing_tagged:
                    with tagged_output_tsv.open("w", encoding="utf-8", newline="") as dst, base_output_tsv.open(
                        "r", encoding="utf-8", newline=""
                    ) as src:
                        dst.write(src.read())
                    generated_names.append(tagged_output_tsv.name)
                print(
                    f"[OK] {json_file.name} -> generated={generated_names}, skipped_existing={skipped_names}, "
                    f"rows={written}, skipped={skipped}"
                )
                base_output_tsv.unlink(missing_ok=True)
            else:
                if base_output_tsv.exists():
                    print(f"[SKIP] {json_file.name} -> {base_output_tsv.name} already exists.")
                    done += 1
                    continue
                written, skipped = convert_one_json(json_file, base_output_tsv, image_root)
                print(f"[OK] {json_file.name} -> {base_output_tsv.name}, rows={written}, skipped={skipped}")
            done += 1
        except Exception as e:
            print(f"[ERROR] {json_file}: {e}")

    print(f"Done. success={done}, failed={len(json_files) - done}")


if __name__ == "__main__":
    main()