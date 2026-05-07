#!/usr/bin/env python3
"""
Classify images under a directory into:
1) chart
2) illustration
3) Photograph
4) Structure
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Tuple

from openai import OpenAI

DEFAULT_IMAGE_ROOT = "./original_data"
DEFAULT_IMAGE_FILE_ROOT = "./original_image"
DEFAULT_CLASSIFIED_ROOT = "./classified_image"
DEFAULT_OUTPUT_PATH = "./classification_result.json"

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

TIMEOUT = 45
MAX_TOKENS = 256
TEMPERATURE = 0.0

VALID_LABELS = {"chart", "illustration", "Photograph", "Structure"}

SYSTEM_PROMPT = """You are an expert scientific-image classifier.
Classify the given image into exactly one label:
- chart: tables, line charts, bar charts, pie charts, or other statistical plots.
- illustration: pipelines, framework diagrams, mind maps, apparatus schematics, conceptual drawings.
- Photograph: camera-captured photos (e.g., microscope photos, satellite photos, natural photos).
- Structure: single chemistry/molecular structure diagrams usually drawable with RDKit.

Return JSON only:
{
  "label": "one of chart|illustration|Photograph|Structure"
}
"""

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class _SimpleProgressBar:
    """Fallback progress bar when tqdm is unavailable."""

    def __init__(self, total: int, desc: str = "Progress", width: int = 30) -> None:
        self.total = max(1, total)
        self.desc = desc
        self.width = width
        self.current = 0
        self._render()

    def _render(self) -> None:
        ratio = self.current / self.total
        done = int(self.width * ratio)
        bar = "#" * done + "-" * (self.width - done)
        msg = f"\r{self.desc}: [{bar}] {self.current}/{self.total}"
        print(msg, end="", flush=True)

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        self._render()
        if self.current >= self.total:
            print()

    def set_postfix_str(self, _: str) -> None:
        return

    def close(self) -> None:
        if self.current < self.total:
            self.current = self.total
            self._render()
            print()


def _build_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output is not valid JSON: {raw!r}") from None
        return json.loads(raw[start : end + 1])


def _to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    content = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{suffix};base64,{content}"


def _is_size_or_format_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("image_too_large" in msg) or ("unsupported image" in msg)


def _build_compatible_image(path: Path, max_bytes: int = 4_900_000) -> Path:
    if Image is None:
        raise RuntimeError("Pillow is required for image compression fallback. Please install pillow.")

    with Image.open(path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        cur_w, cur_h = img.size
        scales = [1.0, 0.85, 0.7, 0.55]
        qualities = [90, 80, 70, 60, 50]

        for scale in scales:
            if scale < 1.0:
                w = max(1, int(cur_w * scale))
                h = max(1, int(cur_h * scale))
                work = img.resize((w, h), Image.Resampling.LANCZOS)
            else:
                work = img
            for quality in qualities:
                fd, tmp_name = tempfile.mkstemp(suffix=".jpeg", prefix="cls_retry_")
                os.close(fd)
                tmp_path = Path(tmp_name)
                work.save(tmp_path, format="JPEG", quality=quality, optimize=True)
                if tmp_path.stat().st_size <= max_bytes:
                    return tmp_path
                tmp_path.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to compress image under {max_bytes} bytes: {path}")


def _iter_relative_image_paths(image_root: Path) -> Iterable[str]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in image_exts:
            yield path.relative_to(image_root).as_posix()


def _iter_relative_image_paths_from_qa(qa_root: Path) -> Iterable[str]:
    seen = set()
    for qa_file in sorted(qa_root.rglob("*_qa.json")):
        try:
            qa_items = json.loads(qa_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(qa_items, list):
            continue
        for item in qa_items:
            if not isinstance(item, dict):
                continue
            for rel_path in item.get("image_path", []):
                if isinstance(rel_path, str):
                    clean = rel_path.strip().replace("\\", "/")
                    if clean and clean not in seen:
                        seen.add(clean)
                        yield clean


def _resolve_image_path(rel_path: str, image_root: Path, image_file_root: Path | None = None) -> Path:
    clean = rel_path.strip().replace("\\", "/")
    if clean.startswith("image/") and image_file_root is not None:
        clean = clean[len("image/") :]
        return image_file_root / clean
    return image_root / clean


def _extract_dataset(rel_path: str) -> str:
    clean = rel_path.strip().replace("\\", "/")
    if clean.startswith("image/"):
        clean = clean[len("image/") :]
    parts = Path(clean).parts
    if not parts:
        return "root"
    # MMMU images are organized as MMMU/<subject>/<file>, keep subject in dataset key.
    if parts[0] == "MMMU" and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _copy_to_class_folder(
    src_abs_path: Path,
    rel_path: str,
    label: str,
    classified_root: Path,
) -> Path:
    dataset = _extract_dataset(rel_path)
    clean = rel_path.strip().replace("\\", "/")
    if clean.startswith("image/"):
        clean = clean[len("image/") :]
    parts = Path(clean).parts
    if parts and parts[0] == "MMMU" and len(parts) >= 3:
        rel_in_dataset = Path(*parts[2:])
    elif len(parts) > 1:
        rel_in_dataset = Path(*parts[1:])
    else:
        rel_in_dataset = Path(src_abs_path.name)
    dst_path = classified_root / label / dataset / rel_in_dataset
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_abs_path, dst_path)
    return dst_path


def classify_image(client: OpenAI, image_path: Path, model: str, timeout: int) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    payload = [
        {"type": "text", "text": "Classify this image using exactly one allowed label."},
        {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=timeout,
        )
    except Exception as exc:
        if not _is_size_or_format_error(exc):
            raise
        tmp_path = _build_compatible_image(image_path)
        try:
            retry_payload = [
                {"type": "text", "text": "Classify this image using exactly one allowed label."},
                {"type": "image_url", "image_url": {"url": _to_data_url(tmp_path)}},
            ]
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": retry_payload},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=timeout,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    raw = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_object(raw)
    label = str(obj.get("label", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label returned for {image_path}: {label!r}")
    return label


def _classify_one(
    rel_path: str,
    image_root: Path,
    image_file_root: Path | None,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> Tuple[str, str]:
    abs_path = _resolve_image_path(rel_path, image_root=image_root, image_file_root=image_file_root)
    client = _build_client(api_key=api_key, base_url=base_url)
    label = classify_image(client, abs_path, model=model, timeout=timeout)
    return rel_path, label


def run_classification(
    image_root: Path,
    image_file_root: Path,
    classified_root: Path,
    output_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    max_workers: int,
    timeout: int,
    resume: bool,
    retry_errors: bool,
) -> Dict[str, str]:
    rel_paths = list(_iter_relative_image_paths(image_root))
    use_qa_mode = len(rel_paths) == 0
    if use_qa_mode:
        rel_paths = list(_iter_relative_image_paths_from_qa(image_root))
    total = len(rel_paths)
    classification_list: Dict[str, str] = {}
    if resume and output_path.exists():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                classification_list.update({str(k): str(v) for k, v in loaded.items()})
                print(f"Resume enabled: loaded {len(classification_list)} existing records from {output_path}")
        except Exception:
            print(f"Resume enabled but failed to parse existing output: {output_path}")

    done_keys = set(classification_list.keys())
    error_keys = {
        k
        for k, v in classification_list.items()
        if isinstance(v, str) and v.startswith("ERROR:")
    }
    if retry_errors:
        pending_rel_paths = [p for p in rel_paths if p in error_keys]
    else:
        pending_rel_paths = [p for p in rel_paths if p not in done_keys]
    total_pending = len(pending_rel_paths)
    if total == 0:
        return classification_list
    if total_pending == 0:
        if retry_errors:
            print("No ERROR entries to retry. Nothing to do.")
        else:
            print("All images already classified. Nothing to do.")
        return classification_list

    # Preflight one image so network/model issues are visible immediately.
    preflight_rel = pending_rel_paths[0]
    preflight_abs = _resolve_image_path(
        preflight_rel,
        image_root=image_root,
        image_file_root=image_file_root if use_qa_mode else None,
    )
    print(f"Preflight check: {preflight_rel}")
    preflight_client = _build_client(api_key=api_key, base_url=base_url)
    _ = classify_image(preflight_client, preflight_abs, model=model, timeout=timeout)
    print("Preflight passed. Starting batch classification...")

    if tqdm is not None:
        progress = tqdm(total=total_pending, desc="Classifying", unit="img")
    else:
        progress = _SimpleProgressBar(total=total_pending, desc="Classifying")

    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _classify_one,
                rel_path,
                image_root,
                image_file_root if use_qa_mode else None,
                model,
                api_key,
                base_url,
                timeout,
            ): rel_path
            for rel_path in pending_rel_paths
        }

        for future in as_completed(futures):
            rel_path = futures[future]
            try:
                _, label = future.result()
                abs_path = _resolve_image_path(
                    rel_path,
                    image_root=image_root,
                    image_file_root=image_file_root if use_qa_mode else None,
                )
                _copy_to_class_folder(
                    src_abs_path=abs_path,
                    rel_path=rel_path,
                    label=label,
                    classified_root=classified_root,
                )
                classification_list[rel_path] = label
                progress.set_postfix_str(f"latest={Path(rel_path).name}:{label}")
            except Exception as exc:  # keep batch running on individual failures
                classification_list[rel_path] = f"ERROR: {exc}"
                progress.set_postfix_str(f"latest={Path(rel_path).name}:ERROR")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(classification_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            progress.update(1)

    progress.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(classification_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return classification_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify all images under a directory into four categories.")
    parser.add_argument("--image-root", type=Path, default=Path(DEFAULT_IMAGE_ROOT))
    parser.add_argument(
        "--image-file-root",
        type=Path,
        default=Path(DEFAULT_IMAGE_FILE_ROOT),
        help="Used when --image-root contains QA json files with image_path like image/xxx.png.",
    )
    parser.add_argument("--classified-root", type=Path, default=Path(DEFAULT_CLASSIFIED_ROOT))
    parser.add_argument("--output-path", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output json (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume and overwrite results from scratch.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Only retry entries whose existing value starts with 'ERROR:'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_classification(
        image_root=args.image_root,
        image_file_root=args.image_file_root,
        classified_root=args.classified_root,
        output_path=args.output_path,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
        timeout=args.timeout,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )
    ok_count = sum(1 for value in result.values() if value in VALID_LABELS)
    print(f"Done. Saved {len(result)} entries to {args.output_path} (ok={ok_count}).")


if __name__ == "__main__":
    main()
