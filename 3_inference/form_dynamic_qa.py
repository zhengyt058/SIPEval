#!/usr/bin/env python3
"""
Convert modified QA result json to generation json format.

Default behavior (SIP_exp naming):
- Input modified:
  ./modified_data/EMMA/chemistry_T1_qa.json
- Input original-for-generate:
  ./3_inference/qa_for_generate/EMMA_chemistry_original.json
- Output:
  ./3_inference/qa_for_generate/EMMA_chemistry_T1.json

Output each sample as:
{
  "id": ...,
  "query": modified_query,
  "target": modified_answer,
  "field": "Chemistry",
  "image_path": [...from original by id...]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODIFIED_INPUT = Path("./modified_data/EMMA/chemistry_T1_qa.json")
DEFAULT_ORIGINAL_INPUT = Path(
    "./3_inference/qa_for_generate/EMMA_chemistry_original.json"
)
DEFAULT_OUTPUT = Path("./3_inference/qa_for_generate/EMMA_chemistry_T1.json")
DEFAULT_CLASSIFICATION_RESULT = Path("./classification_result.json")
DEFAULT_MODIFIED_IMAGE_ROOT = Path("./modified_image")
DEFAULT_IMAGE_QA_ROOT = Path("./3_inference/qa_for_generate")

ACTION_TO_IMAGE_TYPE = {
    "V1": "chart",
    "V2": "illustration",
    "V3": "photograph",
    "V4": "structure",
}

DATASET_TO_FIELD = {
    "EMMA": "chemistry",
    "GEOBench-VLM": "geography",
    "MatCha": "materials",
    "MicroVQA": "biology",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert modified QA json to generation format.")
    parser.add_argument("--modified_input", type=Path, default=DEFAULT_MODIFIED_INPUT, help="Path to modified qa json.")
    parser.add_argument("--original_input", type=Path, default=DEFAULT_ORIGINAL_INPUT, help="Path to original generation json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output json.")
    parser.add_argument(
        "--copy_original",
        action="store_true",
        help="Copy original_input to output in generation format without using modified_input.",
    )
    parser.add_argument(
        "--build_image_type_qa",
        action="store_true",
        help="Build QA files by image type action (V1/V2/V3/V4) from *_original.json files.",
    )
    parser.add_argument(
        "--image_action",
        choices=sorted(ACTION_TO_IMAGE_TYPE.keys()),
        help="Image action to build (V1/V2/V3/V4). Used with --build_image_type_qa.",
    )
    parser.add_argument(
        "--classification_result",
        type=Path,
        default=DEFAULT_CLASSIFICATION_RESULT,
        help="Path to classification result json mapping image path to image type.",
    )
    parser.add_argument(
        "--modified_image_root",
        type=Path,
        default=DEFAULT_MODIFIED_IMAGE_ROOT,
        help="Root of modified images, e.g. /.../SIP_exp/modified_image.",
    )
    parser.add_argument(
        "--image_qa_root",
        type=Path,
        default=DEFAULT_IMAGE_QA_ROOT,
        help="Root containing *_original.json and outputs for image-type QA files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rewrite_image_path(raw_path: str) -> str:
    path_str = str(raw_path)
    if path_str.startswith("image/"):
        return "original_image/" + path_str[len("image/") :]
    return path_str


def normalize_id(value: Any) -> str | None:
    if isinstance(value, (int, str)):
        return str(value)
    return None


def build_image_map(original_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    image_map: dict[str, list[str]] = {}
    for item in original_items:
        sample_id = normalize_id(item.get("id"))
        image_paths = item.get("image_path")
        if sample_id is not None and isinstance(image_paths, list):
            image_map[sample_id] = [rewrite_image_path(str(p)) for p in image_paths]
    return image_map


def convert(modified_data: dict[str, Any], image_map: dict[str, list[str]]) -> list[dict[str, Any]]:
    results = modified_data.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid modified input: key 'results' must be a list.")

    converted: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    skipped_verify_failed = 0
    subject = str(modified_data.get("subject", "unknown"))

    for item in results:
        if not isinstance(item, dict):
            continue

        if item.get("verify_passed") is not True:
            skipped_verify_failed += 1
            continue

        sample_id = normalize_id(item.get("id"))
        if sample_id is None:
            continue

        query = item.get("modified_query")
        answer = item.get("modified_answer")
        if not isinstance(query, str) or not isinstance(answer, str):
            continue

        image_path = image_map.get(sample_id)
        if image_path is None:
            missing_ids.append(sample_id)
            image_path = []

        converted.append(
            {
                "id": sample_id,
                "query": query,
                "target": answer,
                "field": subject,
                "image_path": image_path,
            }
        )

    if missing_ids:
        # Keep conversion successful, but expose potential alignment issue.
        print(f"Warning: missing image_path for ids: {sorted(set(missing_ids), key=str)[:20]} ...")
    if skipped_verify_failed:
        print(f"Skipped {skipped_verify_failed} items with verify_passed != True")

    return converted


def convert_original_only(original_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in original_items:
        if not isinstance(item, dict):
            continue

        sample_id = item.get("id")
        normalized_id = normalize_id(sample_id)
        query = item.get("query")
        target = item.get("target")
        image_path = item.get("image_path")

        if normalized_id is None:
            continue
        if not isinstance(query, str) or target is None:
            continue
        if not isinstance(image_path, list):
            image_path = []

        converted.append(
            {
                "id": normalized_id,
                "query": query,
                "target": str(target),
                "field": str(item.get("field", "Chemistry")),
                "image_path": [rewrite_image_path(str(p)) for p in image_path],
            }
        )
    return converted


def to_modified_image_path(original_image_path: str) -> str:
    path_str = str(original_image_path)
    if path_str.startswith("image/"):
        return "modified_image/" + path_str[len("image/") :]
    if path_str.startswith("original_image/"):
        return "modified_image/" + path_str[len("original_image/") :]
    return path_str


def build_classification_map(raw: dict[str, Any]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        mapped[k] = v.strip().lower()
    return mapped


def convert_original_by_image_type(
    original_items: list[dict[str, Any]],
    classification_map: dict[str, str],
    target_image_type: str,
    modified_image_root: Path,
    default_field: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    missing_classification = 0
    missing_modified_image = 0

    for item in original_items:
        if not isinstance(item, dict):
            continue

        sample_id = normalize_id(item.get("id"))
        query = item.get("query")
        target = item.get("target")
        image_path = item.get("image_path")

        if sample_id is None or not isinstance(query, str) or target is None:
            continue
        if not isinstance(image_path, list) or not image_path:
            continue

        image_key = str(image_path[0])
        if image_key.startswith("original_image/"):
            image_key = "image/" + image_key[len("original_image/") :]

        classified_type = classification_map.get(image_key)
        if classified_type is None:
            missing_classification += 1
            continue
        if classified_type != target_image_type:
            continue

        modified_rel_path = to_modified_image_path(image_key)
        modified_abs_path = (modified_image_root.parent / modified_rel_path).resolve()
        if not modified_abs_path.exists():
            missing_modified_image += 1
            continue

        converted.append(
            {
                "id": sample_id,
                "query": query,
                "target": str(target),
                "field": str(item.get("field", default_field)),
                "image_path": [modified_rel_path],
            }
        )

    if missing_classification:
        print(f"Skipped {missing_classification} items missing classification result")
    if missing_modified_image:
        print(f"Skipped {missing_modified_image} items missing modified image file")
    return converted


def build_image_type_qa_files(args: argparse.Namespace) -> None:
    if not args.image_action:
        raise ValueError("--image_action is required when using --build_image_type_qa")

    classification_raw = load_json(args.classification_result)
    if not isinstance(classification_raw, dict):
        raise ValueError("classification_result root must be a dict.")
    classification_map = build_classification_map(classification_raw)

    target_image_type = ACTION_TO_IMAGE_TYPE[args.image_action]
    original_files = sorted(args.image_qa_root.glob("*_original.json"))
    if not original_files:
        raise FileNotFoundError(f"No *_original.json found under: {args.image_qa_root}")

    for original_file in original_files:
        original_data = load_json(original_file)
        if not isinstance(original_data, list):
            raise ValueError(f"Original input root must be a list: {original_file}")

        original_name = original_file.name
        dataset = original_name.split("_", 1)[0]
        default_field = DATASET_TO_FIELD.get(dataset, "general")

        converted = convert_original_by_image_type(
            original_items=original_data,
            classification_map=classification_map,
            target_image_type=target_image_type,
            modified_image_root=args.modified_image_root,
            default_field=default_field,
        )

        output = original_file.with_name(original_file.stem.replace("_original", f"_{args.image_action}") + ".json")
        if not converted:
            print(f"Converted 0 items, skip writing output: {output}")
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"Converted {len(converted)} items -> {output}")


def main() -> None:
    args = parse_args()

    if args.build_image_type_qa:
        build_image_type_qa_files(args)
        return

    original_data = load_json(args.original_input)

    if not isinstance(original_data, list):
        raise ValueError("Original input root must be a list.")

    if args.copy_original:
        converted = convert_original_only(original_data)
    else:
        modified_data = load_json(args.modified_input)
        if not isinstance(modified_data, dict):
            raise ValueError("Modified input root must be a dict.")
        image_map = build_image_map(original_data)
        converted = convert(modified_data, image_map)

    if not converted:
        print(f"Converted 0 items, skip writing output: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} items -> {args.output}")


if __name__ == "__main__":
    main()