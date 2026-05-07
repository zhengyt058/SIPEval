#!/usr/bin/env python3
"""
V2 / Step-3:
- Read step2 records for one round.
- Verify answer consistency (original vs modified image) in parallel.
- Mark verify_fail/retry_times and emit retry subset for next round.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Dict, List

from openai import OpenAI

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-mYDVzMMHpaUyuSWqwi42JaUs1ZNCBKaYxQuTHNX1siu2wEVG")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.2")
TIMEOUT = 180
MAX_TOKENS = 512
TEMPERATURE = 0.0
DEFAULT_TEMP_DIR = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_image/temp_v2")

SYSTEM_VERIFY = """You are a strict visual QA checker.
You must answer with exactly one word: Yes or No.
Do not output any other text.
"""

_thread_local = local()


def _print_progress(prefix: str, done: int, total: int) -> None:
    total = max(1, total)
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * done / total
    end = "\n" if done >= total else "\r"
    print(f"{prefix} [{bar}] {done}/{total} ({pct:.1f}%)", end=end, flush=True)


def _build_client() -> OpenAI:
    return OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)


def _client() -> OpenAI:
    cli = getattr(_thread_local, "client", None)
    if cli is None:
        cli = _build_client()
        _thread_local.client = cli
    return cli


def _to_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    content = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{ext};base64,{content}"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _delete_modified_image_if_exists(record: Dict[str, Any]) -> None:
    modified_path = record.get("modified_image_path")
    if not modified_path:
        return
    p = Path(str(modified_path))
    if p.exists():
        p.unlink()


def _verify_one(record: Dict[str, Any], model: str) -> Dict[str, Any]:
    prev_retry = int(record.get("retry_times", 0) or 0)
    if not record.get("ok"):
        _delete_modified_image_if_exists(record)
        return {
            **record,
            "verify_fail": True,
            "retry_times": prev_retry + 1,
            "verified_passed": False,
            "verification": {"judge_answer": "Yes", "reason": "previous fail"},
        }

    query = str(record.get("query", ""))
    target = str(record.get("target", ""))
    original_image = Path(str(record["image_path_abs"]))
    modified_image = Path(str(record["modified_image_path"]))

    prompt = (
        "Image-1 is original and Image-2 is modified. "
        f"Question: {query}\nGold answer: {target}\n"
        "Judge whether modification affects answer correctness. Answer only Yes or No."
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_VERIFY},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _to_data_url(original_image)}},
                    {"type": "image_url", "image_url": {"url": _to_data_url(modified_image)}},
                ],
            },
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )
    raw = (resp.choices[0].message.content or "").strip()
    normalized = raw.lower().strip(" .,!?:;\"'")
    same = normalized == "no"
    if not same:
        _delete_modified_image_if_exists(record)
    return {
        **record,
        "text_model_verify": model,
        "verified_passed": bool(same),
        "verify_fail": not bool(same),
        "retry_times": prev_retry if same else (prev_retry + 1),
        "verification": {
            "judge_answer": raw,
            "same_as_target": same,
            "explanation": "No means modification does not affect answer correctness.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Step-3: batch verification with multithreading.")
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--text-model", type=str, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument("--max-retry-times", type=int, default=3)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--retry-output-path", type=Path, default=None)
    parser.add_argument("--final-failed-output-path", type=Path, default=None)
    args = parser.parse_args()

    step2_path = args.input_path or (args.temp_dir / "step2_records.jsonl")
    out_path = args.output_path or (args.temp_dir / "step3_records.jsonl")
    summary_path = args.summary_path or (args.temp_dir / "step3_summary.json")
    retry_output_path = args.retry_output_path or (args.temp_dir / "retry_records.jsonl")
    final_failed_output_path = args.final_failed_output_path or (args.temp_dir / "final_failed_records.jsonl")

    rows = _read_jsonl(step2_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No records in {step2_path}")

    done: List[Dict[str, Any]] = []
    total = len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_verify_one, r, args.text_model): r for r in rows}
        _print_progress("[Step3]", 0, total)
        for i, fut in enumerate(as_completed(futures), start=1):
            src = futures[fut]
            try:
                payload = fut.result()
            except Exception as err:  # noqa: BLE001
                prev_retry = int(src.get("retry_times", 0) or 0)
                payload = {
                    **src,
                    "verify_fail": True,
                    "retry_times": prev_retry + 1,
                    "verified_passed": False,
                    "verification": {"judge_answer": "Yes", "error": str(err)},
                }
            done.append(payload)
            _print_progress("[Step3]", i, total)

    done.sort(key=lambda x: (x.get("dataset", ""), x.get("id", ""), x.get("image_path_rel", "")))
    with out_path.open("w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retry_rows = [x for x in done if x.get("verify_fail") and int(x.get("retry_times", 0) or 0) < args.max_retry_times]
    final_failed_rows = [
        {**x, "final_status": "cannot_modify"}
        for x in done
        if x.get("verify_fail") and int(x.get("retry_times", 0) or 0) >= args.max_retry_times
    ]
    with retry_output_path.open("w", encoding="utf-8") as f:
        for row in retry_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with final_failed_output_path.open("w", encoding="utf-8") as f:
        for row in final_failed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "count_total": len(done),
        "count_verified_passed": sum(1 for x in done if x.get("verified_passed")),
        "count_verified_failed": sum(1 for x in done if not x.get("verified_passed")),
        "count_verify_fail_retryable": len(retry_rows),
        "count_final_cannot_modify": len(final_failed_rows),
        "round_idx": done[0].get("round_idx") if done else None,
        "max_retry_times": args.max_retry_times,
        "text_model_verify": args.text_model,
        "workers": args.workers,
        "step3_records_path": str(out_path),
        "retry_records_path": str(retry_output_path),
        "final_failed_records_path": str(final_failed_output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
