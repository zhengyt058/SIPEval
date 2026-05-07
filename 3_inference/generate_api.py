import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers visual questions.\n"
    "For multiple-choice questions, return only the option letter(s).\n"
    "For yes/no questions, return only Yes or No.\n"
    "Do not provide explanations."
)

DEFAULT_API_BASE_URL = "http://35.220.164.252:3888/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API inference on QA JSON data.")
    parser.add_argument(
        "--output_root",
        default="./3_inference/generate",
        help="Output root folder. Final path: {output_root}/{model}/{dataset}_{action}_generate.json",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name(s), comma separated")
    parser.add_argument("--action", required=True, help="Action name(s), comma separated")
    parser.add_argument("--run_tag", default="", help="Optional run tag for output naming")
    parser.add_argument("--model_path", default="gpt-4.1-mini", help="API model name")
    parser.add_argument(
        "--workplace_path",
        default=".",
        help="Workplace path",
    )
    parser.add_argument(
        "--qa_root",
        default="./3_inference/qa_for_generate",
        help="QA input root path containing files like {dataset}_{action}.json",
    )
    parser.add_argument(
        "--image_root",
        default=".",
        help="Image input root path for resolving relative image paths",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Reserved batch size for API mode")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max output tokens for API")
    parser.add_argument("--greedy", type=str, default="true", help="Use greedy decoding: true/false")
    parser.add_argument("--top_p", type=float, default=0.8, help="Sampling top_p")
    parser.add_argument("--top_k", type=int, default=20, help="Reserved top_k")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Reserved repetition penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--out_seq_length", type=int, default=16384, help="Reserved output sequence length")
    parser.add_argument("--local_files_only", action="store_true", help="Reserved compatibility flag")
    parser.add_argument(
        "--api_base_url",
        default=DEFAULT_API_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="API key (if empty, read from OPENAI_API_KEY)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="API timeout seconds")
    parser.add_argument("--max_workers", type=int, default=12, help="Thread pool workers for API requests")
    parser.add_argument("--max_image_size_mb", type=float, default=1.65, help="Max image size before compress")
    parser.add_argument("--max_dimension", type=int, default=2048, help="Max image dimension after resize")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality when compressing")
    return parser.parse_args()


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_multi_values(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def model_slug(model_path: str) -> str:
    tail = model_path.strip().rstrip("/").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", tail)


def collect_task_files(qa_root: Path, datasets: List[str], actions: List[str]) -> List[Tuple[str, str, Path]]:
    tasks: List[Tuple[str, str, Path]] = []
    missing_files: List[Path] = []
    for dataset in datasets:
        for action in actions:
            input_file = qa_root / f"{dataset}_{action}.json"
            if not input_file.exists():
                missing_files.append(input_file)
                continue
            tasks.append((dataset, action, input_file))
    if missing_files:
        preview = ", ".join(str(p) for p in missing_files[:10])
        more = f" ... (+{len(missing_files) - 10} more)" if len(missing_files) > 10 else ""
        print(f"[Warn] Skip missing input files: {preview}{more}")
    return tasks


def load_items_from_json(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        target = item.get("target")
        image_paths = item.get("image_path", [])
        if not isinstance(image_paths, list):
            continue
        if query is None or target is None or not image_paths:
            continue
        normalized.append(
            {
                "id": item.get("id"),
                "query": str(query),
                "target": target,
                "image_path": image_paths,
                "source_file": str(json_path),
            }
        )
    return normalized


def resolve_image_paths(item: Dict[str, Any], image_root: Path) -> List[Path]:
    resolved: List[Path] = []
    for rel in item["image_path"]:
        rel_path = Path(str(rel))
        resolved.append(rel_path if rel_path.is_absolute() else image_root / rel_path)
    return resolved


def get_mime_type(ext: str) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return mapping.get(ext.lower(), "image/jpeg")


def encode_image_to_base64(image_path: Path, args: argparse.Namespace) -> Tuple[str, str]:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > args.max_dimension:
        img.thumbnail((args.max_dimension, args.max_dimension), Image.Resampling.LANCZOS)

    mime = get_mime_type(image_path.suffix)
    buf = BytesIO()

    raw_size_mb = image_path.stat().st_size / (1024 * 1024)
    if raw_size_mb > args.max_image_size_mb or mime in {"image/tiff", "image/bmp"}:
        img.save(buf, format="JPEG", quality=args.jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        fmt = "PNG" if mime == "image/png" else "JPEG"
        save_kwargs = {"optimize": True}
        if fmt == "JPEG":
            save_kwargs["quality"] = args.jpeg_quality
        img.save(buf, format=fmt, **save_kwargs)
        mime = "image/png" if fmt == "PNG" else "image/jpeg"

    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return base64_str, mime


def load_api_client(args: argparse.Namespace):
    from openai import OpenAI

    api_key = args.api_key or __import__("os").environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set --api_key or OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=args.api_base_url, timeout=args.timeout)


def run_inference_single(client, item: Dict[str, Any], image_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    item_copy = dict(item)
    try:
        content: List[Dict[str, Any]] = []
        for p in resolve_image_paths(item, image_root):
            if not p.exists():
                raise FileNotFoundError(f"image not found: {p}")
            b64, mime = encode_image_to_base64(p, args)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": f"{item['query']}\n\n{DEFAULT_SYSTEM_PROMPT}"})

        messages = [{"role": "user", "content": content}]
        greedy = str_to_bool(args.greedy)
        temperature = 0.0 if greedy else args.temperature
        req = {
            "model": args.model_path,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": temperature,
            "presence_penalty": args.presence_penalty,
        }
        # Some providers (e.g. Bedrock models used by Claude) reject requests
        # when both `temperature` and `top_p` are specified.
        # So we enforce "only one of them" to avoid 400 ValidationException.
        if temperature == 0.0:
            # keep temperature (deterministic), drop top_p
            pass
        else:
            # keep top_p (sampling), drop temperature
            req.pop("temperature", None)
            req["top_p"] = args.top_p

        model_name = str(args.model_path).strip().lower()
        # For qwen3.5 family on this gateway, explicitly disable thinking mode.
        if model_name.startswith("qwen3.5"):
            req["extra_body"] = {"enable_thinking": False}

        response = client.chat.completions.create(**req)

        # Different OpenAI-compatible wrappers may return different response shapes.
        # Prefer OpenAI-like `choices[0].message.content`, but fall back safely.
        out = ""
        if hasattr(response, "choices"):
            out = (response.choices[0].message.content or "").strip()
        elif isinstance(response, dict) and "choices" in response and response["choices"]:
            msg = response["choices"][0].get("message", {})
            out = (msg.get("content") or "").strip()
        elif isinstance(response, str):
            out = response.strip()
        else:
            out = str(response).strip()

        item_copy["output"] = out
        item_copy["status"] = "ok"
        return item_copy
    except Exception as e:
        item_copy["output"] = f"[ERROR] api inference failed: {e}"
        item_copy["status"] = "error"
        return item_copy


def _norm_id(value: Any) -> str:
    return "" if value is None else str(value)


def _result_is_ok(res: Dict[str, Any]) -> bool:
    if not isinstance(res, dict):
        return False
    if res.get("status") != "ok":
        return False
    out = str(res.get("output", ""))
    if "[ERROR]" in out or "Error:" in out or "Exception:" in out:
        return False
    return True


def load_existing_results_map(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    try:
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = data

    if not isinstance(results, list):
        return {}

    mapped: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        rid = _norm_id(r.get("id"))
        if rid:
            mapped[rid] = r
    return mapped


def main() -> None:
    args = parse_args()
    _ = Path(args.workplace_path).expanduser().resolve()
    qa_root = Path(args.qa_root).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    client = load_api_client(args)
    datasets = parse_multi_values(args.dataset)
    actions = parse_multi_values(args.action)
    tasks = collect_task_files(qa_root=qa_root, datasets=datasets, actions=actions)
    if not tasks:
        raise FileNotFoundError("No task found. Provide valid --dataset and --action.")

    model_dir = output_root / model_slug(args.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(args.run_tag or "").strip()
    run_suffix = f"__{run_tag}" if run_tag else ""
    print(f"Found {len(tasks)} task file(s). Output dir: {model_dir}")

    grand_total = 0
    grand_ok = 0
    for dataset, action, input_file in tasks:
        items = load_items_from_json(input_file)
        if not items:
            print(f"[Skip] No valid items in {input_file}")
            continue

        output_path = model_dir / f"{dataset}_{action}{run_suffix}_generate.json"

        existing_map = load_existing_results_map(output_path)
        results_by_id: Dict[str, Dict[str, Any]] = dict(existing_map)

        rerun_items: List[Dict[str, Any]] = []
        for item in items:
            rid = _norm_id(item.get("id"))
            if rid and rid in existing_map and _result_is_ok(existing_map[rid]):
                continue
            rerun_items.append(item)

        pbar = tqdm(total=len(rerun_items), desc=f"{dataset}-{action}")
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_to_item = {
                executor.submit(run_inference_single, client, item, image_root, args): item for item in rerun_items
            }
            for future in as_completed(future_to_item):
                try:
                    res = future.result()
                except Exception as e:
                    item = future_to_item[future]
                    res = {
                        "id": item.get("id"),
                        "source_file": item.get("source_file"),
                        "query": item.get("query"),
                        "target": item.get("target"),
                        "output": f"[ERROR] thread execution failed: {e}",
                        "status": "error",
                    }

                rid = _norm_id(res.get("id"))
                if not rid:
                    pbar.update(1)
                    continue
                results_by_id[rid] = {
                    "id": res.get("id"),
                    "source_file": res.get("source_file"),
                    "query": res.get("query"),
                    "target": res.get("target"),
                    "output": res.get("output"),
                    "status": res.get("status"),
                }
                pbar.update(1)
        pbar.close()

        results: List[Dict[str, Any]] = []
        for item in items:
            rid = _norm_id(item.get("id"))
            r = results_by_id.get(rid)
            if not r:
                r = {
                    "id": item.get("id"),
                    "source_file": item.get("source_file"),
                    "query": item.get("query"),
                    "target": item.get("target"),
                    "output": "[ERROR] missing result after resume",
                    "status": "error",
                }
            results.append(r)

        payload = {
            "dataset": dataset,
            "action": action,
            "run_tag": run_tag,
            "model_path": args.model_path,
            "input_file": str(input_file),
            "output_file": str(output_path),
            "num_samples": len(items),
            "batch_size": args.batch_size,
            "decode_config": {
                "greedy": str_to_bool(args.greedy),
                "top_p": args.top_p,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "presence_penalty": args.presence_penalty,
                "out_seq_length": args.out_seq_length,
                "max_new_tokens": args.max_new_tokens,
            },
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        ok_count = sum(1 for x in results if x.get("status") == "ok")
        grand_total += len(results)
        grand_ok += ok_count
        print(f"Saved {len(results)} results to: {output_path} (ok={ok_count}, error={len(results) - ok_count})")

    print(f"All done. files={len(tasks)}, total_results={grand_total}, ok={grand_ok}, error={grand_total - grand_ok}")


if __name__ == "__main__":
    main()
