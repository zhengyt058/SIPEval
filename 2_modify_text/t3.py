#!/usr/bin/env python3
"""
T3：科学背景扰动（Wikipedia 引导 + LLM 生成）

支持：
1) 单条：--query / --query-file + --target
2) 批量：--dataset + --input_path + --output_path（输入输出逻辑对齐 t2.py）

每条结果字段（按需求）：
id, can_modify, original_query, original_answer, modified_query, modified_answer,
reason, return_checklist（wikipedia 检索词及其返回内容）, generated_background, verify_passed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import wikipedia
from openai import OpenAI
from tqdm import tqdm

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-mYDVzMMHpaUyuSWqwi42JaUs1ZNCBKaYxQuTHNX1siu2wEVG")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
TIMEOUT = 120
MAX_TOKENS = 2048
TEMPERATURE = 0.0
DEFAULT_MAX_WORKERS = 1
MAX_WIKI_KEYWORDS = 2

DEFAULT_WIKI_SENTENCES = int(os.getenv("T3_WIKI_SENTENCES", "20"))
DEFAULT_TARGET_WORDS = int(os.getenv("T3_TARGET_WORDS", "1200"))
WIKI_HTTP_TIMEOUT_SEC = int(os.getenv("T3_WIKI_TIMEOUT_SEC", "10"))
WIKI_MAX_RETRIES = int(os.getenv("T3_WIKI_MAX_RETRIES", "2"))
WIKI_FETCH_BUDGET_SEC = float(os.getenv("T3_WIKI_BUDGET_SEC", "25"))
WIKI_MAX_TERM_SEC = float(os.getenv("T3_WIKI_MAX_TERM_SEC", "12"))
WIKI_API_BASE = "https://en.wikipedia.org/w/api.php"
WIKI_CACHE_PATH = Path(__file__).with_name("t3_wiki_summary_cache.json")
WIKI_CACHE_LOCK = threading.Lock()
WIKI_USER_AGENT = "SIP-T3/1.0 (Wikipedia REST summary cache)"
PERF_LOG_ENABLED = os.getenv("T3_PERF_LOG", "1").strip().lower() not in {"0", "false", "no"}
ALLOW_NO_WIKI_FALLBACK = os.getenv("T3_ALLOW_NO_WIKI_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}

DEFAULT_INPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data")
DEFAULT_OUTPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_data")

client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
_thread_local = threading.local()
wikipedia.set_lang("en")
def _load_wiki_cache() -> Dict[str, Dict[str, str]]:
    if not WIKI_CACHE_PATH.exists():
        return {}
    try:
        with WIKI_CACHE_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            out: Dict[str, Dict[str, str]] = {}
            for k, v in obj.items():
                if not isinstance(v, dict):
                    continue
                title = str(v.get("title", "") or "").strip()
                summary = str(v.get("summary", "") or "").strip()
                if k and title and summary:
                    out[k] = {"title": title, "summary": summary}
            return out
    except Exception:
        return {}
    return {}


_WIKI_SUMMARY_CACHE: Dict[str, Dict[str, str]] = _load_wiki_cache()
_WIKI_CACHE_DIRTY = False


def _save_wiki_cache() -> None:
    WIKI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = WIKI_CACHE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_WIKI_SUMMARY_CACHE, f, ensure_ascii=False, indent=2)
    tmp.replace(WIKI_CACHE_PATH)


def _cache_key(term: str) -> str:
    return " ".join((term or "").split()).strip().casefold()


def _get_cached_wiki_summary(term: str) -> Dict[str, str] | None:
    key = _cache_key(term)
    if not key:
        return None
    with WIKI_CACHE_LOCK:
        return _WIKI_SUMMARY_CACHE.get(key)


def _put_cached_wiki_summary(term: str, title: str, summary: str) -> None:
    global _WIKI_CACHE_DIRTY
    key = _cache_key(term)
    if not key or not title or not summary:
        return
    with WIKI_CACHE_LOCK:
        _WIKI_SUMMARY_CACHE[key] = {"title": title, "summary": summary}
        _WIKI_CACHE_DIRTY = True


def _flush_wiki_cache_if_dirty() -> None:
    global _WIKI_CACHE_DIRTY
    with WIKI_CACHE_LOCK:
        if not _WIKI_CACHE_DIRTY:
            return
        _save_wiki_cache()
        _WIKI_CACHE_DIRTY = False


def _fetch_wikipedia_summary_rest(term: str, sentences: int, *, term_timeout_sec: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "query_term": term,
        "title": None,
        "summary": None,
        "error": None,
        "source": "rest",
    }
    last_err = ""
    timeout_s = max(3.0, min(float(term_timeout_sec), float(WIKI_HTTP_TIMEOUT_SEC)))

    def _simple_fetch() -> Dict[str, str]:
        summary = wikipedia.summary(
            term,
            sentences=max(1, int(sentences)),
            auto_suggest=True,
        )
        page = wikipedia.page(term, auto_suggest=True)
        return {"title": page.title, "summary": (summary or "").strip()}

    for attempt in range(1, WIKI_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_simple_fetch)
                payload = fut.result(timeout=timeout_s)
            if not payload["summary"]:
                raise ValueError("Empty summary")
            result["title"] = payload["title"]
            result["summary"] = payload["summary"]
            result["error"] = None
            return result
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e!s}"
            if attempt < WIKI_MAX_RETRIES:
                time.sleep(0.5 * attempt)
    result["error"] = f"Wikipedia fetch failed after {WIKI_MAX_RETRIES} tries: {last_err}"
    return result


SYSTEM_KEYWORDS = """You plan supplementary *encyclopedic* scientific background for a multiple-choice exam item.
Output ONLY valid JSON.

Goal: pick English Wikipedia search phrases that would retrieve text useful as *lengthy* background in front of the question (definitions, conventions, historical context), without trivially solving the item.

If the item cannot support such background (purely visual with no named concepts, nothing sensible to look up, etc.):
  {"can_modify": false, "reason": "short Chinese explanation", "wiki_search_keywords": []}

If it can:
  {"can_modify": true, "reason": "short Chinese note or empty string", "wiki_search_keywords": ["...", ...]}
  - 1–2 distinct English phrases suitable for English Wikipedia (article-title style when possible).
  - Never return more than 2 keywords.
  - Prefer concepts central to understanding the stem (e.g. named rules, scales, mechanisms).
  - No non-English keywords except standard proper names used on English Wikipedia.

Never set can_modify=true with an empty wiki_search_keywords."""


SYSTEM_BACKGROUND = """You write a single English *background* paragraph for a science exam.
Output ONLY valid JSON:
{"background": "...", "approx_word_count": <integer>}

Rules:
1) Ground the prose mainly in <<<WIKI_PARAGRAPH>>>; you may connect it to the exam stem using standard textbook knowledge, but do not invent obscure facts not supported by the paragraph or common science.
2) Target length ≈ TARGET_WORDS English words (±15%). Count roughly; set approx_word_count accordingly.
3) Do NOT reveal or hint which multiple-choice option is correct; do not restate the option texts as rankings or answers.
4) Do NOT include <image ...> placeholders in the background.
5) Keep tone neutral and encyclopedic; one or two short paragraphs as a single string is OK if total word count still matches TARGET_WORDS."""

SYSTEM_BACKGROUND_NO_WIKI = """You write a single English *background* paragraph for a science exam.
Output ONLY valid JSON:
{"background": "...", "approx_word_count": <integer>}

Rules:
1) Wikipedia retrieval failed. Build a concise, generic textbook-style background using only the query topic and common science knowledge.
2) Target length ≈ TARGET_WORDS English words (±15%).
3) Do NOT reveal or hint which multiple-choice option is correct.
4) Do NOT include <image ...> placeholders in the background.
5) Keep it high-level and safe: definitions, context, terminology, and common mechanisms only."""


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output is not valid JSON: {text!r}") from None
        return json.loads(text[start : end + 1])


def _chat_json(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> dict:
    active = client_override if client_override is not None else client
    resp = active.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _extract_json_object(raw)


def _extract_image_tokens(text: str) -> List[str]:
    return re.findall(r"<image\s+\d+>", text or "", flags=re.IGNORECASE)


def _word_count_en(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))


def _perf_log(stage: str, elapsed_sec: float, extra: str = "") -> None:
    if not PERF_LOG_ENABLED:
        return
    suffix = f" | {extra}" if extra else ""
    print(f"[T3 PERF] {stage}: {elapsed_sec:.2f}s{suffix}", flush=True)


def judge_wiki_keywords(
    query: str,
    target: str,
    *,
    model: str,
    client_override: OpenAI | None = None,
    temperature: float,
) -> Tuple[bool, str, List[str]]:
    user = (
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        'Output exactly: {"can_modify": true/false, "reason": "...", "wiki_search_keywords": ["...", ...]}'
    )
    obj = _chat_json(
        SYSTEM_KEYWORDS,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    can_modify = bool(obj.get("can_modify", False))
    reason = str(obj.get("reason", "") or "").strip()
    raw = obj.get("wiki_search_keywords") or obj.get("returned_checklist") or []
    out: List[str] = []
    if isinstance(raw, list):
        for x in raw:
            s = str(x).strip()
            if s and s not in out:
                out.append(s)
    out = out[:MAX_WIKI_KEYWORDS]
    if not can_modify:
        return False, reason, []
    if not out:
        return False, (reason + "；" if reason else "") + "未给出检索词。", []
    return True, reason, out


def fetch_wiki_summary_blocks(
    keywords: List[str],
    *,
    sentences: int,
    budget_sec: float = WIKI_FETCH_BUDGET_SEC,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    deadline = time.monotonic() + max(1.0, float(budget_sec))
    for kw in keywords:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append(
                {
                    "query_term": kw,
                    "title": None,
                    "summary": None,
                    "error": f"Skipped: Wikipedia budget exceeded ({budget_sec:.0f}s)",
                    "source": "budget_skip",
                }
            )
            continue
        cached = _get_cached_wiki_summary(kw)
        if cached:
            results.append(
                {
                    "query_term": kw,
                    "title": cached["title"],
                    "summary": cached["summary"],
                    "error": None,
                    "source": "cache",
                }
            )
            continue
        fetched = _fetch_wikipedia_summary_rest(
            kw,
            sentences=sentences,
            term_timeout_sec=min(WIKI_MAX_TERM_SEC, remaining),
        )
        if fetched.get("summary"):
            _put_cached_wiki_summary(
                kw,
                str(fetched.get("title") or ""),
                str(fetched.get("summary") or ""),
            )
            results.append(fetched)
            # Fast path: one valid wiki block is enough for background generation.
            break
        results.append(fetched)
    return results


def build_wiki_paragraph(wiki_blocks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for b in wiki_blocks:
        summ = (b.get("summary") or "").strip()
        title = b.get("title") or b.get("query_term")
        if summ:
            parts.append(f"({title}): {summ}")
            break
    return "\n\n".join(parts)


def _verify_by_wiki_fetch(wiki_blocks: List[Dict[str, Any]]) -> bool:
    """Pass if not all Wikipedia lookups failed."""
    return any(bool((b.get("summary") or "").strip()) for b in wiki_blocks)


def generate_background_paragraph(
    wiki_paragraph: str,
    query: str,
    target: str,
    target_words: int,
    *,
    model: str,
    client_override: OpenAI | None = None,
    temperature: float,
) -> Tuple[str, int]:
    user = (
        f"TARGET_WORDS = {target_words}\n\n"
        "<<<WIKI_PARAGRAPH>>>\n"
        f"{wiki_paragraph}\n"
        "<<<WIKI_PARAGRAPH_END>>>\n\n"
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        'Output exactly one JSON object: {"background": "...", "approx_word_count": <int>}'
    )
    obj = _chat_json(
        SYSTEM_BACKGROUND,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    bg = str(obj.get("background", "") or "").strip()
    approx = int(obj.get("approx_word_count", 0) or 0)
    if not bg:
        raise ValueError("Model returned empty background")
    wc = _word_count_en(bg)
    if approx <= 0:
        approx = wc
    return bg, approx


def generate_background_without_wiki(
    query: str,
    target: str,
    target_words: int,
    keywords: List[str],
    *,
    model: str,
    client_override: OpenAI | None = None,
    temperature: float,
) -> Tuple[str, int]:
    user = (
        f"TARGET_WORDS = {target_words}\n\n"
        "<<<KEYWORDS>>>\n"
        f"{', '.join(keywords)}\n"
        "<<<KEYWORDS_END>>>\n\n"
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        'Output exactly one JSON object: {"background": "...", "approx_word_count": <int>}'
    )
    obj = _chat_json(
        SYSTEM_BACKGROUND_NO_WIKI,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    bg = str(obj.get("background", "") or "").strip()
    approx = int(obj.get("approx_word_count", 0) or 0)
    if not bg:
        raise ValueError("Model returned empty fallback background")
    wc = _word_count_en(bg)
    if approx <= 0:
        approx = wc
    return bg, approx


def prepend_background(background: str, query: str) -> str:
    bg = background.strip()
    q = query.strip()
    if not bg:
        return q
    return bg + "\n\n" + q


def run_pipeline_once(
    query: str,
    target: str,
    *,
    model: str = DEFAULT_MODEL,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
    wiki_sentences: int = DEFAULT_WIKI_SENTENCES,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    can_modify, reason, keywords = judge_wiki_keywords(
        query,
        target,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    _perf_log("judge_wiki_keywords", time.perf_counter() - t0, f"can_modify={can_modify}, keywords={len(keywords)}")
    if not can_modify:
        _perf_log("run_pipeline_once_total", time.perf_counter() - t_all, "early_exit=not_modifiable")
        return {
            "can_modify": False,
            "original_query": query,
            "original_answer": target,
            "modified_query": query,
            "modified_answer": target,
            "reason": reason,
            "generated_background": "",
            "return_checklist": {"wiki_search_keywords": [], "wikipedia_results": []},
            "verify_passed": None,
        }

    t0 = time.perf_counter()
    wiki_blocks = fetch_wiki_summary_blocks(keywords, sentences=wiki_sentences)
    _perf_log("fetch_wiki_summary_blocks", time.perf_counter() - t0, f"keywords={len(keywords)}")
    verify_passed = _verify_by_wiki_fetch(wiki_blocks)
    wiki_para = build_wiki_paragraph(wiki_blocks)
    if not wiki_para.strip():
        if not ALLOW_NO_WIKI_FALLBACK:
            _perf_log("run_pipeline_once_total", time.perf_counter() - t_all, "early_exit=no_wiki_summary")
            return {
                "can_modify": True,
                "original_query": query,
                "original_answer": target,
                "modified_query": query,
                "modified_answer": target,
                "reason": (reason + "；" if reason else "") + "Wikipedia 未取到有效摘要，跳过背景生成。",
                "generated_background": "",
                "return_checklist": {
                    "wiki_search_keywords": keywords,
                    "wikipedia_results": wiki_blocks,
                },
                "verify_passed": verify_passed,
            }

        t0 = time.perf_counter()
        bg_text, _approx = generate_background_without_wiki(
            query=query,
            target=target,
            target_words=target_words,
            keywords=keywords,
            model=model,
            client_override=client_override,
            temperature=temperature,
        )
        _perf_log("generate_background_no_wiki", time.perf_counter() - t0, f"target_words={target_words}")
        modified = prepend_background(bg_text, query)
        _perf_log("run_pipeline_once_total", time.perf_counter() - t_all, "fallback=no_wiki")
        return {
            "can_modify": True,
            "original_query": query,
            "original_answer": target,
            "modified_query": modified,
            "modified_answer": target,
            "reason": (reason + "；" if reason else "") + "Wikipedia 未命中，使用 query-only 背景兜底生成。",
            "generated_background": bg_text,
            "return_checklist": {
                "wiki_search_keywords": keywords,
                "wikipedia_results": wiki_blocks,
                "fallback_mode": "no_wiki",
                "word_count_en_tokens": _word_count_en(bg_text),
            },
            "verify_passed": True,
        }

    t0 = time.perf_counter()
    bg_text, _approx = generate_background_paragraph(
        wiki_para,
        query,
        target,
        target_words,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    _perf_log("generate_background_paragraph", time.perf_counter() - t0, f"target_words={target_words}")
    modified = prepend_background(bg_text, query)
    _perf_log("run_pipeline_once_total", time.perf_counter() - t_all)

    return {
        "can_modify": True,
        "original_query": query,
        "original_answer": target,
        "modified_query": modified,
        "modified_answer": target,
        "reason": reason,
        "generated_background": bg_text,
        "return_checklist": {
            "wiki_search_keywords": keywords,
            "wikipedia_results": wiki_blocks,
            "word_count_en_tokens": _word_count_en(bg_text),
        },
        "verify_passed": verify_passed,
    }


def resolve_input_file(dataset: str, subject: str, input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    qa_file = input_path / dataset / f"{subject}_qa.json"
    if not qa_file.exists():
        raise FileNotFoundError(f"Input file not found: {qa_file}")
    return qa_file


def resolve_output_file(dataset: str, subject: str, action: str, output_path: Path) -> Path:
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    ds_dir = output_path / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir / f"{subject}_{action}_qa.json"


def parse_multi_values(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _normalize_for_checklist_match(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = " ".join(t.split())
    return t.casefold()


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        key = _normalize_for_checklist_match(x)
        if not key or key in seen:
            continue
        out.append(x)
        seen.add(key)
    return out


def discover_dataset_subject_map(input_path: Path) -> Dict[str, List[str]]:
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input root dir not found: {input_path}")

    dataset_subjects: Dict[str, List[str]] = {}
    for dataset_dir in sorted([p for p in input_path.iterdir() if p.is_dir()]):
        subjects: List[str] = []
        for file_path in sorted(dataset_dir.glob("*_qa.json")):
            name = file_path.stem
            if name.endswith("_qa"):
                subject = name[: -len("_qa")].strip()
                if subject:
                    subjects.append(subject)
        if subjects:
            dataset_subjects[dataset_dir.name] = _dedup_keep_order(subjects)
    return dataset_subjects


def get_thread_client(api_key: str, base_url: str) -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            timeout=TIMEOUT,
        )
    return _thread_local.client


def process_one_item(
    item: Dict[str, Any],
    model: str,
    temperature: float,
    wiki_sentences: int,
    target_words: int,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    item_id = item.get("id")
    original_query = str(item.get("query", "") or "")
    original_target = str(item.get("target", "") or "")
    try:
        thread_client = get_thread_client(api_key=api_key, base_url=base_url)
        result = run_pipeline_once(
            original_query,
            original_target,
            model=model,
            client_override=thread_client,
            temperature=temperature,
            wiki_sentences=wiki_sentences,
            target_words=target_words,
        )
        return {
            "id": item_id,
            "can_modify": result["can_modify"],
            "original_query": result["original_query"],
            "original_answer": result["original_answer"],
            "modified_query": result["modified_query"],
            "modified_answer": result["modified_answer"],
            "reason": result.get("reason", ""),
            "return_checklist": result.get("return_checklist", {}),
            "generated_background": result.get("generated_background", ""),
            "verify_passed": result.get("verify_passed"),
        }
    except Exception as e:
        return {
            "id": item_id,
            "can_modify": False,
            "original_query": original_query,
            "original_answer": original_target,
            "modified_query": original_query,
            "modified_answer": original_target,
            "reason": f"Error: {str(e)}",
            "return_checklist": {"wiki_search_keywords": [], "wikipedia_results": []},
            "generated_background": "",
            "verify_passed": False,
        }


def dynamic_modify_queries(
    dataset: str,
    subject: str,
    action: str,
    input_path: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    wiki_sentences: int = DEFAULT_WIKI_SENTENCES,
    target_words: int = DEFAULT_TARGET_WORDS,
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    in_file = resolve_input_file(dataset, subject, Path(input_path))
    out_file = resolve_output_file(dataset, subject, action, Path(output_path))
    with in_file.open("r", encoding="utf-8") as f:
        qa_items = json.load(f)

    modified_items: List[Dict[str, Any]] = [None] * len(qa_items)
    run_indices = list(range(len(qa_items)))

    # 断点续跑：若输出已存在，仅重跑 verify_passed=False 的条目
    if out_file.exists():
        try:
            with out_file.open("r", encoding="utf-8") as f:
                old_summary = json.load(f)
            old_results = old_summary.get("results", [])
            if isinstance(old_results, list) and len(old_results) == len(qa_items):
                for idx, old_item in enumerate(old_results):
                    if isinstance(old_item, dict):
                        modified_items[idx] = old_item
                run_indices = [
                    idx
                    for idx, old_item in enumerate(old_results)
                    if (not isinstance(old_item, dict))
                    or (old_item.get("verify_passed") is False)
                ]
                print(
                    f"Resume mode: found existing output, reprocessing only failed items: "
                    f"{len(run_indices)}/{len(qa_items)}"
                )
            else:
                print("Resume mode skipped: existing output format/length mismatch, full run.")
        except Exception as e:
            print(f"Resume mode skipped: failed to read existing output ({e}), full run.")

    if not run_indices:
        print("No failed items to reprocess. Rewriting summary from existing results.")
    
    safe_workers = 1
    with ThreadPoolExecutor(max_workers=safe_workers) as executor:
        future_to_index = {
            executor.submit(
                process_one_item,
                qa_items[idx],
                model,
                temperature,
                wiki_sentences,
                target_words,
                api_key,
                base_url,
            ): idx
            for idx in run_indices
        }
        with tqdm(total=len(run_indices), desc=f"{dataset}-{action}", unit="item") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                modified_items[idx] = future.result()
                pbar.update(1)

    changed = sum(1 for x in modified_items if x.get("can_modify"))
    unchanged = len(modified_items) - changed
    verify_ok = sum(1 for x in modified_items if x.get("verify_passed") is True)
    verify_fail = sum(1 for x in modified_items if x.get("verify_passed") is False)
    verify_na = sum(1 for x in modified_items if x.get("verify_passed") is None)

    summary = {
        "dataset": dataset,
        "subject": subject,
        "action": action,
        "input_file": str(in_file),
        "output_file": str(out_file),
        "total": len(modified_items),
        "changed": changed,
        "unchanged": unchanged,
        "verify_passed": verify_ok,
        "verify_failed": verify_fail,
        "verify_not_applicable": verify_na,
        "results": modified_items,
    }
    _flush_wiki_cache_if_dirty()
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="T3 background augmentation (single or batch).")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Batch mode: dataset name(s), comma separated, e.g. MMMU,MatCha, or 'all'.",
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Batch mode: process all datasets discovered under --input_path.",
    )
    parser.add_argument(
        "--subject",
        default="all",
        help="Batch mode: subject name(s), comma separated, e.g. chemistry,biology, or 'all'.",
    )
    parser.add_argument("--action", default="T3", help="Batch mode: action id(s), comma separated.")
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH), help="Batch mode input file or root dir.")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH), help="Batch mode output file or dir.")
    parser.add_argument("--query", type=str, default=None, help="Single mode full query.")
    parser.add_argument("--query-file", type=Path, default=None, help="Single mode UTF-8 query file.")
    parser.add_argument("--target", "-a", type=str, default=None, help="Single mode answer.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--wiki-sentences", type=int, default=DEFAULT_WIKI_SENTENCES)
    parser.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", "-o", type=Path, default=None, help="Single mode output JSON path.")
    args = parser.parse_args()

    if args.dataset or args.all_datasets:
        input_root = Path(args.input_path)
        dataset_subject_map = discover_dataset_subject_map(input_root)
        if not dataset_subject_map:
            raise FileNotFoundError(
                f"No '*_qa.json' files found under input root: {input_root}"
            )

        if args.all_datasets or (args.dataset and args.dataset.strip().lower() == "all"):
            datasets = list(dataset_subject_map.keys())
        else:
            if not args.dataset:
                parser.error("Batch mode needs --dataset (or set --all_datasets).")
            datasets = parse_multi_values(args.dataset)
            missing_datasets = [d for d in datasets if d not in dataset_subject_map]
            if missing_datasets:
                parser.error(
                    "Unknown dataset(s): "
                    + ",".join(missing_datasets)
                    + f". Available: {','.join(dataset_subject_map.keys())}"
                )

        subject_all = args.subject.strip().lower() == "all"
        selected_subjects = [] if subject_all else parse_multi_values(args.subject)
        if not subject_all and not selected_subjects:
            parser.error("Batch mode needs --subject, or set --subject all.")

        actions = parse_multi_values(args.action)
        all_summaries: List[Dict[str, Any]] = []
        pairs: List[Tuple[str, str]] = []
        for dataset in datasets:
            available_subjects = dataset_subject_map.get(dataset, [])
            if subject_all:
                chosen = available_subjects
            else:
                chosen = [s for s in selected_subjects if s in available_subjects]
                missing_subjects = [s for s in selected_subjects if s not in available_subjects]
                if missing_subjects:
                    print(
                        f"Skip dataset={dataset}; missing subjects={missing_subjects}; "
                        f"available={available_subjects}"
                    )
            for subject in chosen:
                pairs.append((dataset, subject))

        if not pairs:
            parser.error("No dataset-subject tasks to run after filtering.")

        total_cases = len(pairs) * len(actions)
        print(
            f"Will run {total_cases} task(s): "
            f"dataset-subject pairs={pairs}, actions={actions}"
        )
        for dataset, subject in pairs:
            for action in actions:
                print(f"\nRunning dataset={dataset}, subject={subject}, action={action}")
                summary = dynamic_modify_queries(
                    dataset=dataset,
                    subject=subject,
                    action=action,
                    input_path=args.input_path,
                    output_path=args.output_path,
                    model=args.model,
                    temperature=args.temperature,
                    wiki_sentences=args.wiki_sentences,
                    target_words=args.target_words,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    max_workers=args.max_workers,
                )
                all_summaries.append(summary)
                print(
                    f"Done. total={summary['total']}, changed={summary['changed']}, "
                    f"unchanged={summary['unchanged']}, verify_failed={summary['verify_failed']}\n"
                    f"output={summary['output_file']}"
                )

        grand_total = sum(x["total"] for x in all_summaries)
        grand_changed = sum(x["changed"] for x in all_summaries)
        grand_unchanged = sum(x["unchanged"] for x in all_summaries)
        grand_verify_failed = sum(x["verify_failed"] for x in all_summaries)
        print(
            f"\nAll finished. files={len(all_summaries)}, total={grand_total}, "
            f"changed={grand_changed}, unchanged={grand_unchanged}, "
            f"verify_failed={grand_verify_failed}"
        )
        return

    if args.query_file is not None:
        query = args.query_file.read_text(encoding="utf-8")
    elif args.query is not None:
        query = args.query
    else:
        parser.error("Provide --query / --query-file in single mode, or use --dataset for batch mode.")

    if args.target is None:
        parser.error("--target / -a is required in single mode.")

    result = run_pipeline_once(
        query,
        args.target,
        model=args.model,
        temperature=args.temperature,
        wiki_sentences=args.wiki_sentences,
        target_words=args.target_words,
    )
    result_single = {
        "id": "single_test",
        "can_modify": result["can_modify"],
        "original_query": result["original_query"],
        "original_answer": result["original_answer"],
        "modified_query": result["modified_query"],
        "modified_answer": result["modified_answer"],
        "reason": result.get("reason", ""),
        "return_checklist": result.get("return_checklist", {}),
        "generated_background": result.get("generated_background", ""),
        "verify_passed": result.get("verify_passed"),
    }
    print(json.dumps(result_single, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result_single, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote to {args.output}")


if __name__ == "__main__":
    main()
