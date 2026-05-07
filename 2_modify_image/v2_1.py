#!/usr/bin/env python3
"""
V2 / Step-1:
- Round-1: batch-read all QA files from original_data.
- Round-2/3: read retry subset from previous step3 retry file.
- Generate edit plans with GPT in parallel.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-mYDVzMMHpaUyuSWqwi42JaUs1ZNCBKaYxQuTHNX1siu2wEVG")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.2")
TIMEOUT = 180
MAX_TOKENS = 1024
TEMPERATURE = 0.0

DEFAULT_QA_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data")
DEFAULT_IMAGE_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_image")
DEFAULT_CLASSIFICATION = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/classification_result.json")
DEFAULT_TEMP_DIR = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_image/temp_v2")

ImageKind = Literal["illustration", "photograph"]
_thread_local = local()

SYSTEM_SUGGEST_ILLUSTRATION = """You are an assistant for robust visual QA testing.
Given question, gold answer, and image:
- Propose 1-3 short irrelevant TEXT annotations that can be added to the image.
- They must not alter key evidence required to solve the question.
- Place text in blank/background areas only.

Return JSON only:
{
  "return_list": [
    {"text":"...", "position":"top-left|top-right|bottom-left|bottom-right|top-center|bottom-center"}
  ],
  "reason": "why these are irrelevant"
}"""

SYSTEM_SUGGEST_PHOTOGRAPH = """You are an assistant for robust visual QA testing on real-looking images.
Given question, gold answer, and image:
- Propose 1-3 irrelevant NOISE- or ARTIFACT-like visual additions that fit the real-world modality of the image.
- Descriptions must be plausible for the scene type; do not add readable text labels as the main edit.
- Choose positions in areas that do NOT occlude regions needed to answer the question.

For each item, "text" is a short English phrase describing WHAT to add,
and "position" is one of:
top-left|top-right|bottom-left|bottom-right|top-center|bottom-center

Return JSON only:
{
  "return_list": [
    {"text":"...", "position":"top-left|top-right|bottom-left|bottom-right|top-center|bottom-center"}
  ],
  "reason": "why these do not affect answering"
}"""


def _build_client() -> OpenAI:
    return OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)


def _client() -> OpenAI:
    cli = getattr(_thread_local, "client", None)
    if cli is None:
        cli = _build_client()
        _thread_local.client = cli
    return cli


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        left = raw.find("{")
        right = raw.rfind("}")
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Model output is not valid JSON: {raw!r}") from None
        return json.loads(raw[left : right + 1])


def _to_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    content = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{ext};base64,{content}"


def _normalize_kind(raw_kind: str) -> Optional[ImageKind]:
    key = (raw_kind or "").strip().lower()
    if key in {"photograph", "photo"}:
        return "photograph"
    if key in {"illustration"}:
        return "illustration"
    return None


def _load_classification_map(path: Path) -> Dict[str, ImageKind]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"classification file must be object: {path}")
    out: Dict[str, ImageKind] = {}
    for rel, label in data.items():
        if not isinstance(rel, str):
            continue
        norm = _normalize_kind(str(label))
        if norm is None:
            continue
        out[rel] = norm
    return out


def _load_records(
    qa_root: Path, image_root: Path, cls_map: Dict[str, ImageKind]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "count_total_qa_items": 0,
        "count_missing_image_file": 0,
        "count_missing_or_non_target_classification": 0,
        "count_selected": 0,
    }
    for qa_path in sorted(qa_root.glob("*/*_qa.json")):
        dataset = qa_path.parent.name
        qa_items = json.loads(qa_path.read_text(encoding="utf-8"))
        if not isinstance(qa_items, list):
            continue
        for item in qa_items:
            stats["count_total_qa_items"] += 1
            if not isinstance(item, dict):
                continue
            rel_images = item.get("image_path") or []
            if not isinstance(rel_images, list) or not rel_images:
                continue
            rel_image = str(rel_images[0])  # e.g. image/MMMU/biology/1.png
            rel_from_root = rel_image.removeprefix("image/")
            src_image = image_root / rel_from_root
            if not src_image.exists():
                stats["count_missing_image_file"] += 1
                continue
            kind = cls_map.get(rel_image, cls_map.get(f"image/{rel_from_root}"))
            if kind is None:
                stats["count_missing_or_non_target_classification"] += 1
                continue
            records.append(
                {
                    "dataset": dataset,
                    "qa_file": str(qa_path),
                    "id": str(item.get("id", "")),
                    "query": str(item.get("query", "")),
                    "target": str(item.get("target", "")),
                    "image_path_rel": rel_from_root,
                    "image_path_abs": str(src_image),
                    "image_type": kind,
                    "retry_times": 0,
                    "verify_fail": False,
                }
            )
    stats["count_selected"] = len(records)
    return records, stats


def _load_retry_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        rows.append(item)
    return rows


def _propose_for_record(record: Dict[str, Any], model: str, round_idx: int) -> Dict[str, Any]:
    image_path = Path(record["image_path_abs"])
    image_kind: ImageKind = record["image_type"]
    system = SYSTEM_SUGGEST_PHOTOGRAPH if image_kind == "photograph" else SYSTEM_SUGGEST_ILLUSTRATION
    user = (
        f"Question:\n{record['query']}\n\nGold answer:\n{record['target']}\n\n"
        "Return JSON only."
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
                ],
            },
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )
    raw = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_object(raw)
    items = obj.get("return_list") or []
    cleaned: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items[:3]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            position = str(it.get("position", "")).strip()
            if text and position:
                cleaned.append({"text": text, "position": position})

    return {
        **record,
        "text_model": model,
        "return_list": cleaned,
        "reason": str(obj.get("reason", "") or ""),
        "round_idx": round_idx,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Step-1: batch GPT proposal with multithreading.")
    parser.add_argument("--qa-root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--text-model", type=str, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument("--round-idx", type=int, default=1)
    parser.add_argument("--input-records", type=Path, default=None, help="Retry subset jsonl from previous step3.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    if args.input_records is not None:
        records = _load_retry_records(args.input_records)
    else:
        cls_map = _load_classification_map(args.classification)
        records, load_stats = _load_records(args.qa_root, args.image_root, cls_map)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No valid records found for Step1 input.")

    out_path = args.output_path or (args.temp_dir / "step1_records.jsonl")
    summary_path = args.summary_path or (args.temp_dir / "step1_summary.json")

    done: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_propose_for_record, r, args.text_model, args.round_idx): r for r in records}
        for i, fut in enumerate(as_completed(futures), start=1):
            src = futures[fut]
            try:
                payload = fut.result()
                if not payload.get("return_list"):
                    payload["ok"] = False
                    payload["error"] = "empty return_list"
            except Exception as err:  # noqa: BLE001
                payload = {**src, "round_idx": args.round_idx, "ok": False, "return_list": [], "reason": "", "error": str(err)}
            done.append(payload)
            if i % 50 == 0 or i == len(records):
                print(f"[Step1] processed {i}/{len(records)}")

    done.sort(key=lambda x: (x.get("dataset", ""), x.get("id", ""), x.get("image_path_rel", "")))
    with out_path.open("w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "count_total": len(done),
        "count_ok": sum(1 for x in done if x.get("ok")),
        "count_failed": sum(1 for x in done if not x.get("ok")),
        "round_idx": args.round_idx,
        "text_model": args.text_model,
        "workers": args.workers,
        "step1_records_path": str(out_path),
    }
    if args.input_records is None:
        summary.update(load_stats)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
