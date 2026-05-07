#!/usr/bin/env python3
"""
Build composite QA json: samples that appear in both a T-level file and a V-level file
under qa_for_generate.

For each id in the intersection:
- query, target, field: from the T file (text-side modification)
- image_path: from the V file (visual-side modification)

Example (T3 + V4):
  Input:  EMMA_chemistry_T3.json, EMMA_chemistry_V4.json
  Output: EMMA_chemistry_T3V4.json

Naming: {dataset}_{field}_{T_tag}{V_tag}.json  e.g. T3 and V4 -> ..._T3V4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QA_ROOT = Path(
    "./3_inference/qa_for_generate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge T-level and V-level qa_for_generate json by intersecting ids."
    )
    parser.add_argument(
        "-T",
        "--text_mod",
        default="T3",
        help="Text modification tag in filenames, e.g. T1, T2, T3, T4 (default: T3).",
    )
    parser.add_argument(
        "-V",
        "--visual_mod",
        default="V4",
        help="Visual modification tag in filenames, e.g. V1, V2, V3, V4 (default: V4).",
    )
    parser.add_argument(
        "--qa_root",
        type=Path,
        default=DEFAULT_QA_ROOT,
        help="Directory containing *_T*.json and *_V*.json generation files.",
    )
    parser.add_argument(
        "--single_pair",
        nargs=2,
        metavar=("T_JSON", "V_JSON"),
        type=Path,
        help="Optional: merge exactly one T file and one V file; output path follows composite naming next to them.",
    )
    parser.add_argument(
        "--build_t2t4",
        action="store_true",
        help=(
            "Build *_T2T4.json text-only files under qa_root by combining T2 stem "
            "+ T4 options (id intersection), without any visual merge."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_id(value: Any) -> str | None:
    if isinstance(value, (int, str)):
        return str(value)
    return None


def parse_name_parts(stem: str) -> tuple[str, str, str] | None:
    """
    Parse stem like 'EMMA_chemistry_T3' or 'GEOBench-VLM_geography_V4'
    into (dataset, field, level_tag).
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    level = parts[-1]
    field = parts[-2]
    dataset = "_".join(parts[:-2])
    return dataset, field, level


def items_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = normalize_id(item.get("id"))
        if sid is None:
            continue
        out[sid] = item
    return out


def split_stem_and_options(query: str) -> tuple[str, str]:
    """
    Split a multiple-choice query into (stem, options_block).

    We treat lines starting with "<capital>." as options (A., B., ...).
    Everything before the first such line is considered the stem.
    """
    lines = query.split("\n")
    first_opt_idx = None
    for i, line in enumerate(lines):
        # e.g. "A. xxx", "B. yyy"
        if len(line) >= 2 and line[0].isalpha() and line[1] == ".":
            first_opt_idx = i
            break
    if first_opt_idx is None:
        return query, ""
    stem = "\n".join(lines[:first_opt_idx])
    options_block = "\n".join(lines[first_opt_idx:])
    return stem, options_block


def build_t2t4_for_pair(
    t2_items: Iterable[dict[str, Any]],
    t4_items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Construct T2T4 composite items:
    - stem (question text, including image token) from T2
    - options block from T4
    - target from T4 (since options come from T4)
    - field and image_path copied from T2
    Only ids that appear in both T2 and T4 are kept.
    """
    t4_map = items_by_id(list(t4_items))
    out: list[dict[str, Any]] = []
    for item in t2_items:
        if not isinstance(item, dict):
            continue
        sid = normalize_id(item.get("id"))
        if sid is None:
            continue
        t4_item = t4_map.get(sid)
        if t4_item is None:
            continue

        q2 = item.get("query")
        q4 = t4_item.get("query")
        if not isinstance(q2, str) or not isinstance(q4, str):
            continue

        stem2, _ = split_stem_and_options(q2)
        _, opts4 = split_stem_and_options(q4)

        # Fallbacks: if options not detected in T4, just use its full query.
        if not stem2:
            stem2 = q2
        if not opts4:
            opts4 = q4

        combined_query = stem2
        if opts4:
            combined_query = stem2 + "\n" + opts4

        field = item.get("field")
        image_path = item.get("image_path")
        if not isinstance(image_path, list):
            image_path = []

        target4 = t4_item.get("target")
        if target4 is None:
            continue

        out.append(
            {
                "id": sid,
                "query": combined_query,
                "target": str(target4),
                "field": str(field) if field is not None else "",
                "image_path": [str(p) for p in image_path],
            }
        )
    return out


def merge_pair(
    t_items: list[dict[str, Any]],
    v_items: list[dict[str, Any]],
    t_tag: str,
    v_tag: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Returns (merged_list, n_mismatch_target, n_missing_image_path).
    Order follows the order of appearance in t_items.
    """
    v_map = items_by_id(v_items)
    merged: list[dict[str, Any]] = []
    n_mismatch_target = 0
    n_missing_image_path = 0

    for item in t_items:
        if not isinstance(item, dict):
            continue
        sid = normalize_id(item.get("id"))
        if sid is None:
            continue
        v_item = v_map.get(sid)
        if v_item is None:
            continue

        query = item.get("query")
        target = item.get("target")
        field = item.get("field")
        if not isinstance(query, str) or target is None:
            continue

        v_target = v_item.get("target")
        if v_target is not None and str(v_target) != str(target):
            n_mismatch_target += 1

        image_path = v_item.get("image_path")
        if not isinstance(image_path, list) or not image_path:
            n_missing_image_path += 1
            image_path = []

        merged.append(
            {
                "id": sid,
                "query": query,
                "target": str(target),
                "field": str(field) if field is not None else "",
                "image_path": [str(p) for p in image_path],
            }
        )

    return merged, n_mismatch_target, n_missing_image_path


def composite_stem(dataset: str, field: str, t_tag: str, v_tag: str) -> str:
    return f"{dataset}_{field}_{t_tag}{v_tag}"


def t_text_stem(dataset: str, field: str, t_tag: str) -> str:
    return f"{dataset}_{field}_{t_tag}"


def build_t2t4_batch(qa_root: Path) -> None:
    """
    Build text-only T2T4 JSON files for every dataset/field that has both T2 and T4:
    - Input:  *_T2.json and *_T4.json under qa_root
    - Output: *_T2T4.json under qa_root
    """
    if not qa_root.is_dir():
        raise NotADirectoryError(f"qa_root is not a directory: {qa_root}")

    t2_files = sorted(qa_root.glob("*_T2.json"))
    if not t2_files:
        raise FileNotFoundError(f"No *_T2.json files under {qa_root}")

    for t2_path in t2_files:
        parsed = parse_name_parts(t2_path.stem)
        if parsed is None:
            print(f"[T2T4] Skip (unparseable stem): {t2_path.name}")
            continue
        dataset, field, level = parsed
        if level != "T2":
            continue

        t4_path = qa_root / f"{dataset}_{field}_T4.json"
        if not t4_path.exists():
            print(
                f"[T2T4] No paired T4 file for {t2_path.name}, skip -> {t4_path.name}"
            )
            continue

        out_stem = t_text_stem(dataset, field, "T2T4")
        out_path = qa_root / f"{out_stem}.json"
        if out_path.exists():
            print(f"[T2T4] Output already exists, skip: {out_path.name}")
            continue

        t2_data = load_json(t2_path)
        t4_data = load_json(t4_path)
        if not isinstance(t2_data, list) or not isinstance(t4_data, list):
            print(
                f"[T2T4] Skip (root must be list): {t2_path.name} or {t4_path.name}"
            )
            continue

        merged = build_t2t4_for_pair(t2_data, t4_data)
        if not merged:
            print(f"[T2T4] Intersection empty, skip writing: {out_path}")
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[T2T4] Built {len(merged)} items -> {out_path}")


def process_batch(qa_root: Path, t_tag: str, v_tag: str) -> None:
    if not qa_root.is_dir():
        raise NotADirectoryError(f"qa_root is not a directory: {qa_root}")

    t_suffix = f"_{t_tag}.json"
    v_suffix = f"_{v_tag}.json"

    t_files = sorted(qa_root.glob(f"*{t_suffix}"))
    if not t_files:
        raise FileNotFoundError(f"No *{t_suffix} files under {qa_root}")

    for t_path in t_files:
        parsed = parse_name_parts(t_path.stem)
        if parsed is None:
            print(f"Skip (unparseable stem): {t_path.name}")
            continue
        dataset, field, level_in_file = parsed
        if level_in_file != t_tag:
            continue

        v_path = qa_root / f"{dataset}_{field}_{v_tag}.json"
        if not v_path.exists():
            print(f"No paired {v_tag} file for {t_path.name}, skip -> {v_path.name}")
            continue

        t_data = load_json(t_path)
        v_data = load_json(v_path)
        if not isinstance(t_data, list) or not isinstance(v_data, list):
            print(f"Skip (root must be list): {t_path} or {v_path}")
            continue

        merged, n_tgt_mismatch, n_missing_img = merge_pair(t_data, v_data, t_tag, v_tag)
        out_name = composite_stem(dataset, field, t_tag, v_tag) + ".json"
        out_path = qa_root / out_name

        if not merged:
            print(f"Intersection empty, skip writing: {out_path}")
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        extra = []
        if n_tgt_mismatch:
            extra.append(f"target_mismatch_vs_{v_tag}={n_tgt_mismatch}")
        if n_missing_img:
            extra.append(f"empty_image_path={n_missing_img}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        print(f"Merged {len(merged)} ids -> {out_path}{suffix}")


def process_single_pair(t_path: Path, v_path: Path, t_tag: str, v_tag: str) -> None:
    t_parsed = parse_name_parts(t_path.stem)
    v_parsed = parse_name_parts(v_path.stem)
    if t_parsed is None or v_parsed is None:
        raise ValueError(f"Cannot parse stems: {t_path.stem}, {v_path.stem}")
    d_t, f_t, lt = t_parsed
    d_v, f_v, lv = v_parsed
    if (d_t, f_t) != (d_v, f_v):
        raise ValueError(f"Dataset/field mismatch: {t_path} vs {v_path}")
    if lt != t_tag:
        raise ValueError(f"{t_path.name} level is {lt}, expected {t_tag}")
    if lv != v_tag:
        raise ValueError(f"{v_path.name} level is {lv}, expected {v_tag}")

    t_data = load_json(t_path)
    v_data = load_json(v_path)
    if not isinstance(t_data, list) or not isinstance(v_data, list):
        raise ValueError("Both inputs must be JSON arrays.")

    merged, n_tgt_mismatch, n_missing_img = merge_pair(t_data, v_data, t_tag, v_tag)
    out_path = t_path.parent / (composite_stem(d_t, f_t, t_tag, v_tag) + ".json")
    if not merged:
        print(f"Intersection empty, skip writing: {out_path}")
        return
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    extra = []
    if n_tgt_mismatch:
        extra.append(f"target_mismatch_vs_{v_tag}={n_tgt_mismatch}")
    if n_missing_img:
        extra.append(f"empty_image_path={n_missing_img}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    print(f"Merged {len(merged)} ids -> {out_path}{suffix}")


def main() -> None:
    args = parse_args()
    if args.build_t2t4:
        build_t2t4_batch(args.qa_root)
        return
    if args.single_pair:
        process_single_pair(args.single_pair[0], args.single_pair[1], args.text_mod, args.visual_mod)
    else:
        process_batch(args.qa_root, args.text_mod, args.visual_mod)


if __name__ == "__main__":
    main()
