#!/usr/bin/env python3
"""
V2 / Step-2:
- Read step1 records from temp_v2 (usually one round's subset).
- Edit images with Qwen-Image-Edit.
- Save modified images under modified_image with the SAME relative structure as original_image.
- Write step2 records.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

QWEN_MODEL_PATH = "/mnt/shared-storage-user/zhengyuting/tos/zhengyuting/Models/Qwen-Image-Edit-2511"
DEFAULT_IMAGE_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_image")
DEFAULT_OUTPUT_ROOT = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_image")
DEFAULT_TEMP_DIR = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_image/temp_v2")


def _print_progress(prefix: str, done: int, total: int) -> None:
    total = max(1, total)
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * done / total
    end = "\n" if done >= total else "\r"
    print(f"{prefix} [{bar}] {done}/{total} ({pct:.1f}%)", end=end, flush=True)


def _build_prompt(return_list: List[Dict[str, str]], image_type: str) -> str:
    bullet_lines = [f'- "{x["text"]}" at {x["position"]}' for x in return_list if x.get("text") and x.get("position")]
    if not bullet_lines:
        raise ValueError("return_list is empty")
    if (image_type or "illustration").strip().lower() == "photograph":
        return (
            "Apply subtle realistic visual edits as described below (noise, smudges, cloud-like occlusion, "
            "sensor dust, etc.). Do not cover regions needed to answer the question.\n"
            "Edits:\n" + "\n".join(bullet_lines)
        )
    return (
        "Add these short text notes to blank/background areas only, avoid occluding key evidence.\n"
        "Keep font small and neutral.\n"
        "Notes:\n" + "\n".join(bullet_lines)
    )


def _load_pipeline(model_path: str, device: str):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _run_qwen_edit(pipe: Any, image_path: Path, prompt: str, output_image: Path, seed: int) -> None:
    import torch

    img = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        output = pipe(
            image=[img],
            prompt=prompt,
            generator=torch.manual_seed(seed),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=40,
            guidance_scale=1.0,
            num_images_per_prompt=1,
        )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output.images[0].save(output_image)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _process_one(
    record: Dict[str, Any],
    image_root: Path,
    output_root: Path,
    model_path: str,
    seed: int,
    pipe: Any,
) -> Dict[str, Any]:
    if not record.get("ok"):
        return {**record, "ok": False, "error_step2": "skip because step1 failed"}

    src_abs = Path(record["image_path_abs"])
    rel = record.get("image_path_rel")
    if not rel:
        rel = str(src_abs.relative_to(image_root))
    dst_abs = output_root / rel

    prompt = _build_prompt(record.get("return_list") or [], str(record.get("image_type", "illustration")))
    _run_qwen_edit(pipe, src_abs, prompt, dst_abs, seed)

    return {
        **record,
        "qwen_model_path": model_path,
        "edit_prompt": prompt,
        "modified_image_path": str(dst_abs),
        "seed": seed,
        "ok": True,
    }


def _worker_run(
    worker_idx: int,
    gpu_id: str,
    chunk: List[Dict[str, Any]],
    image_root: Path,
    output_root: Path,
    model_path: str,
    base_seed: int,
    q: Any,
) -> None:
    try:
        device = f"cuda:{gpu_id}"
        pipe = _load_pipeline(model_path, device)
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(chunk):
            try:
                payload = _process_one(
                    row,
                    image_root=image_root,
                    output_root=output_root,
                    model_path=model_path,
                    seed=base_seed + worker_idx * 100000 + i,
                    pipe=pipe,
                )
            except Exception as err:  # noqa: BLE001
                payload = {**row, "ok": False, "error_step2": str(err)}
            out.append(payload)
            q.put({"type": "progress", "count": 1, "worker_idx": worker_idx})
        q.put({"type": "done", "ok": True, "rows": out, "worker_idx": worker_idx})
    except Exception as err:  # noqa: BLE001
        q.put({"type": "done", "ok": False, "rows": [], "worker_idx": worker_idx, "error": str(err)})


def _split_rows(rows: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        chunks[i % n].append(row)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Step-2: batch Qwen edit.")
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--qwen-model-path", type=str, default=QWEN_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="Ignored in multi-gpu mode.")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help='Comma-separated GPU IDs, e.g. "0,1,2,3". One process per GPU.',
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    step1_path = args.input_path or (args.temp_dir / "step1_records.jsonl")
    out_path = args.output_path or (args.temp_dir / "step2_records.jsonl")
    summary_path = args.summary_path or (args.temp_dir / "step2_summary.json")
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(step1_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No records in {step1_path}")

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids is empty")

    chunks = _split_rows(rows, len(gpu_ids))
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_run,
            args=(i, gpu_id, chunks[i], args.image_root, args.output_root, args.qwen_model_path, args.seed, q),
        )
        p.start()
        procs.append(p)

    done: List[Dict[str, Any]] = []
    workers_done = 0
    processed = 0
    total = len(rows)
    _print_progress("[Step2]", processed, total)
    while workers_done < len(procs):
        msg = q.get()
        if msg.get("type") == "progress":
            processed += int(msg.get("count", 0) or 0)
            _print_progress("[Step2]", min(processed, total), total)
            continue
        workers_done += 1
        if not msg.get("ok"):
            print(f"\n[Step2] worker {msg.get('worker_idx')} failed: {msg.get('error')}", flush=True)
        done.extend(msg.get("rows") or [])
        print(f"\n[Step2] worker {msg.get('worker_idx')} returned {len(msg.get('rows') or [])} rows", flush=True)

    for p in procs:
        p.join()

    done.sort(key=lambda x: (x.get("dataset", ""), x.get("id", ""), x.get("image_path_rel", "")))
    with out_path.open("w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "count_total": len(done),
        "count_ok": sum(1 for x in done if x.get("ok")),
        "count_failed": sum(1 for x in done if not x.get("ok")),
        "round_idx": done[0].get("round_idx") if done else None,
        "gpu_ids": gpu_ids,
        "output_root": str(args.output_root),
        "step2_records_path": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
