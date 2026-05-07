import argparse
import json
import os
import random
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import wikipedia
from openai import OpenAI
from tqdm import tqdm

DEFAULT_API_KEY = "sk-mYDVzMMHpaUyuSWqwi42JaUs1ZNCBKaYxQuTHNX1siu2wEVG"
DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"
DEFAULT_MODEL = "gpt-5.2"
TIMEOUT = 120
MAX_TOKENS = 2048
TEMPERATURE = 0.0
DEFAULT_MAX_WORKERS = 8
DEFAULT_INPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data")
DEFAULT_OUTPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_data")
DISTRACTOR_COUNT = 10
WIKI_SENTENCES = 3

client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
_thread_local = threading.local()
wikipedia.set_lang("en")

# wikipedia 包内部 BeautifulSoup(html) 未指定 parser，会触发 GuessedAtParserWarning
try:
    from bs4 import GuessedAtParserWarning

    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass

SYSTEM_JUDGE = """You are a strict science MCQ editor. Output ONLY valid JSON.
Decide whether the question can be modified by adding many irrelevant distractors.

Set can_modify=false when:
- It is not a real multiple-choice question.
- The choice space is inherently closed/finite and cannot be reasonably expanded.

Set can_modify=true only when distractor expansion is reasonable.

Return exactly:
{
  "can_modify": true/false,
  "reason": "brief English explanation"
}
"""

SYSTEM_WIKI_TERMS = """You are a science retrieval assistant. Output ONLY valid JSON.
Given a multiple-choice science question, provide 1-3 English Wikipedia search keywords
that are relevant enough to produce factual background text.

Return exactly:
{
  "keywords": ["term1", "term2"],
  "reason": "brief English explanation"
}
"""

SYSTEM_DISTRACTORS = """You are a science exam distractor writer. Output ONLY valid JSON.
Given the question, options, correct answer label, and Wikipedia snippets, generate exactly 10 distractor options.
Rules:
- Keep style close to original options.
- Must be incorrect choices for this question.
- Avoid duplicates with existing options.
- Do not output labels, only option texts.

Return exactly:
{
  "distractors": ["...", "..."]
}
"""


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


def _extract_option_label(target: str) -> str:
    s = (target or "").strip().upper()
    m = re.fullmatch(r"([A-Z])(?:\b|[\.。,:; ].*)?", s)
    return m.group(1) if m else ""


def _parse_options_dot(query: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in (query or "").splitlines():
        m = re.match(r"^\s*([A-Z])\.\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _parse_options_paren(query: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    text = query or ""
    marks = list(re.finditer(r"\(([A-Z])\)\s*", text))
    if len(marks) < 2:
        return out
    for idx, m in enumerate(marks):
        label = m.group(1)
        start = m.end()
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(text)
        value = text[start:end].strip()
        if value:
            out[label] = value
    return out


def _split_stem_and_option_block(query: str) -> Tuple[str, str]:
    query = (query or "").strip()
    m = re.search(r"(?m)^\s*[A-Z]\.\s+", query)
    if m:
        return query[: m.start()].strip(), query[m.start() :].strip()
    m2 = re.search(r"\([A-Z]\)\s*", query)
    if m2:
        return query[: m2.start()].strip(), query[m2.start() :].strip()
    return query, ""


def parse_mcq(query: str) -> Tuple[bool, str, Dict[str, str]]:
    stem, _ = _split_stem_and_option_block(query)
    options = _parse_options_dot(query)
    if len(options) < 2:
        options = _parse_options_paren(query)
    return (len(options) >= 2), stem, options


def get_thread_client(api_key: str, base_url: str) -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            timeout=TIMEOUT,
        )
    return _thread_local.client


def judge_can_modify(
    query: str,
    target: str,
    options: Dict[str, str],
    *,
    model: str,
    client_override: OpenAI | None,
    temperature: float,
) -> Tuple[bool, str]:
    user = (
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        "<<<OPTIONS_JSON>>>\n"
        f"{json.dumps(options, ensure_ascii=False)}\n"
        "<<<OPTIONS_JSON_END>>>"
    )
    obj = _chat_json(
        SYSTEM_JUDGE,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    return bool(obj.get("can_modify", False)), str(obj.get("reason", "") or "").strip()


def extract_wiki_keywords(
    query: str,
    target: str,
    options: Dict[str, str],
    *,
    model: str,
    client_override: OpenAI | None,
    temperature: float,
) -> Tuple[List[str], str]:
    user = (
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        "<<<OPTIONS_JSON>>>\n"
        f"{json.dumps(options, ensure_ascii=False)}\n"
        "<<<OPTIONS_JSON_END>>>"
    )
    obj = _chat_json(
        SYSTEM_WIKI_TERMS,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    kws: List[str] = []
    for x in (obj.get("keywords") or []):
        s = str(x).strip()
        if s and s not in kws:
            kws.append(s)
    return kws[:3], str(obj.get("reason", "") or "").strip()


def fetch_wikipedia_for_keywords(keywords: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    checklist: List[Dict[str, Any]] = []
    any_success = False
    for kw in keywords:
        entry: Dict[str, Any] = {
            "search_keyword": kw,
            "wikipedia_title": None,
            "wikipedia_summary": None,
            "error": None,
        }
        try:
            try:
                page = wikipedia.page(kw, auto_suggest=False)
            except Exception:
                page = wikipedia.page(kw, auto_suggest=True)
            summary = wikipedia.summary(page.title, sentences=WIKI_SENTENCES, auto_suggest=False)
            entry["wikipedia_title"] = page.title
            entry["wikipedia_summary"] = summary
            if summary and str(summary).strip():
                any_success = True
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
        checklist.append(entry)
        time.sleep(0.2)
    return checklist, any_success


def _build_wiki_context(checklist: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for item in checklist:
        summ = str(item.get("wikipedia_summary") or "").strip()
        if not summ:
            continue
        title = str(item.get("wikipedia_title") or item.get("search_keyword") or "").strip()
        chunks.append(f"- {title}: {summ}")
    return "\n".join(chunks)


def generate_distractors(
    query: str,
    target: str,
    options: Dict[str, str],
    wiki_checklist: List[Dict[str, Any]],
    *,
    model: str,
    client_override: OpenAI | None,
    temperature: float,
) -> List[str]:
    wiki_context = _build_wiki_context(wiki_checklist)
    user = (
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<TARGET>>>\n"
        f"{target}\n"
        "<<<TARGET_END>>>\n\n"
        "<<<OPTIONS_JSON>>>\n"
        f"{json.dumps(options, ensure_ascii=False)}\n"
        "<<<OPTIONS_JSON_END>>>\n\n"
        "<<<WIKIPEDIA_SNIPPETS>>>\n"
        f"{wiki_context}\n"
        "<<<WIKIPEDIA_SNIPPETS_END>>>"
    )
    obj = _chat_json(
        SYSTEM_DISTRACTORS,
        user,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    out: List[str] = []
    existing = {v.strip().casefold() for v in options.values()}
    for x in (obj.get("distractors") or []):
        s = str(x).strip()
        if not s:
            continue
        if s.casefold() in existing:
            continue
        if s.casefold() in {y.casefold() for y in out}:
            continue
        out.append(s)
    return out[:DISTRACTOR_COUNT]


def _format_options(labels: List[str], option_texts: List[str]) -> str:
    lines: List[str] = []
    for i, t in enumerate(option_texts):
        lines.append(f"{labels[i]}. {t}")
    return "\n".join(lines)


def build_modified_mcq(
    original_query: str,
    options: Dict[str, str],
    target: str,
    distractors: List[str],
) -> Tuple[str, str]:
    stem, _ = _split_stem_and_option_block(original_query)
    target_label = _extract_option_label(target)
    if not target_label or target_label not in options:
        return original_query, target

    correct_text = options[target_label].strip()
    wrong_original = [options[k].strip() for k in sorted(options.keys()) if k != target_label]
    all_option_texts = wrong_original + distractors + [correct_text]

    labels = [chr(ord("A") + i) for i in range(len(all_option_texts))]
    random.shuffle(all_option_texts)
    new_target = labels[all_option_texts.index(correct_text)]
    option_block = _format_options(labels, all_option_texts)
    modified_query = (stem.strip() + "\n" + option_block).strip()
    return modified_query, new_target


def run_pipeline_once(
    query: str,
    target: str,
    *,
    model: str = DEFAULT_MODEL,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    is_mcq, _stem, options = parse_mcq(query)
    if not is_mcq:
        return {
            "can_modify": False,
            "original_query": query,
            "original_answer": target,
            "modified_query": query,
            "modified_answer": target,
            "reason": "Not a multiple-choice question.",
            "return_checklist": [],
            "distractors": [],
            "verify_passed": None,
        }

    can_modify, judge_reason = judge_can_modify(
        query,
        target,
        options,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    if not can_modify:
        return {
            "can_modify": False,
            "original_query": query,
            "original_answer": target,
            "modified_query": query,
            "modified_answer": target,
            "reason": judge_reason or "Question judged not expandable.",
            "return_checklist": [],
            "distractors": [],
            "verify_passed": None,
        }

    keywords, kw_reason = extract_wiki_keywords(
        query,
        target,
        options,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    if not keywords:
        return {
            "can_modify": False,
            "original_query": query,
            "original_answer": target,
            "modified_query": query,
            "modified_answer": target,
            "reason": kw_reason or "No Wikipedia search keywords extracted.",
            "return_checklist": [],
            "distractors": [],
            "verify_passed": False,
        }

    return_checklist, any_wiki_success = fetch_wikipedia_for_keywords(keywords)
    distractors = generate_distractors(
        query,
        target,
        options,
        return_checklist,
        model=model,
        client_override=client_override,
        temperature=temperature,
    )
    modified_query, modified_answer = build_modified_mcq(query, options, target, distractors)

    reason_parts = [x for x in [judge_reason, kw_reason] if x]
    if not any_wiki_success:
        reason_parts.append("All Wikipedia fetches failed.")
    if len(distractors) < DISTRACTOR_COUNT:
        reason_parts.append(f"Generated distractors={len(distractors)} < {DISTRACTOR_COUNT}.")

    return {
        "can_modify": True,
        "original_query": query,
        "original_answer": target,
        "modified_query": modified_query,
        "modified_answer": modified_answer,
        "reason": " | ".join(reason_parts),
        "return_checklist": return_checklist,
        "distractors": distractors,
        "verify_passed": any_wiki_success,
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


def process_one_item(
    item: Dict[str, Any],
    model: str,
    temperature: float,
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
        )
        return {
            "id": item_id,
            "can_modify": result["can_modify"],
            "original_query": result["original_query"],
            "original_answer": result["original_answer"],
            "modified_query": result["modified_query"],
            "modified_answer": result["modified_answer"],
            "reason": result.get("reason", ""),
            "return_checklist": result.get("return_checklist", []),
            "distractors": result.get("distractors", []),
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
            "return_checklist": [],
            "distractors": [],
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
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[str, Any]:
    in_file = resolve_input_file(dataset, subject, Path(input_path))
    out_file = resolve_output_file(dataset, subject, action, Path(output_path))

    with in_file.open("r", encoding="utf-8") as f:
        qa_items = json.load(f)

    modified_items: List[Dict[str, Any]] = [None] * len(qa_items)
    safe_workers = max(1, int(max_workers))

    with ThreadPoolExecutor(max_workers=safe_workers) as executor:
        future_to_index = {
            executor.submit(
                process_one_item,
                item,
                model,
                temperature,
                api_key,
                base_url,
            ): idx
            for idx, item in enumerate(qa_items)
        }
        with tqdm(total=len(qa_items), desc=f"{dataset}-{action}", unit="item") as pbar:
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
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="T4 irrelevant distractor expansion (single or batch).")
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
    parser.add_argument("--action", default="T4", help="Batch mode: action id(s), comma separated.")
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH), help="Batch mode input file or root dir.")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH), help="Batch mode output file or dir.")
    parser.add_argument("--query", type=str, default=None, help="Single mode full query.")
    parser.add_argument("--query-file", type=Path, default=None, help="Single mode UTF-8 query file.")
    parser.add_argument("--target", "-a", type=str, default=None, help="Single mode answer.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
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
    )
    result_single = {
        "id": "single_test",
        "can_modify": result["can_modify"],
        "original_query": result["original_query"],
        "original_answer": result["original_answer"],
        "modified_query": result["modified_query"],
        "modified_answer": result["modified_answer"],
        "reason": result.get("reason", ""),
        "return_checklist": result.get("return_checklist", []),
        "distractors": result.get("distractors", []),
        "verify_passed": result.get("verify_passed"),
    }
    print(json.dumps(result_single, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result_single, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote to {args.output}")


if __name__ == "__main__":
    main()
