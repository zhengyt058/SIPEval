import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

SYSTEM_PROMPT = (
    "You rewrite scientific/visual QA query text while keeping semantics and answer unchanged. "
    "Return JSON only."
)


client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
_thread_local = threading.local()


def parse_query(query: str) -> Tuple[str, str]:
    """
    Split full MCQ string into stem (`query_text`) and options block (`options`).
    Handles lines like 'A. ...', '(A)', indented ' A. ', etc.
    """
    query = (query or "").strip()
    if not query:
        return "", ""

    # Leftmost start of a typical first-option line
    patterns = [
        r"(?m)^\s*[A-E]\.\s+",  # A. / B. ...
        r"(?m)^\s*\([A-E]\)\s*[\.\)]?\s*",  # (A) or (A).
        r"(?m)^\s+[A-E]\.\s+",  # leading space then A.
    ]
    best_pos: int | None = None
    for pat in patterns:
        m = re.search(pat, query)
        if m:
            pos = m.start()
            if best_pos is None or pos < best_pos:
                best_pos = pos

    if best_pos is not None:
        query_text = query[:best_pos].strip()
        options = query[best_pos:].strip()
        return query_text, options

    return query, ""


def _normalize_return_checklist(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in s.splitlines() if line.strip()]


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


def parse_output(raw: str) -> Tuple[bool, str, str, List[str]]:
    """
    Parse model JSON: can_modify, modified_query_text, reason, return_checklist.
    Accepts `modified_query` as alias for modified_query_text.
    """
    obj = _extract_json_object(raw)
    can_modify = bool(obj.get("can_modify", False))
    modified_query_text = str(
        obj.get("modified_query_text") or obj.get("modified_query") or ""
    ).strip()
    reason = str(obj.get("reason", "") or "").strip()
    return_checklist = _normalize_return_checklist(obj.get("return_checklist"))
    return can_modify, modified_query_text, reason, return_checklist


def _normalize_for_checklist_match(text: str) -> str:
    """Normalize whitespace and case for stable substring checks."""
    if not text:
        return ""
    t = str(text)
    t = " ".join(t.split())
    return t.casefold()


def _checklist_item_in_query(modified_query_text: str, item: str) -> bool:
    hay = _normalize_for_checklist_match(modified_query_text)
    needle = _normalize_for_checklist_match(item)
    if not needle:
        return True
    return needle in hay


def _is_literal_substring(item: str, source_text: str) -> bool:
    if not item:
        return False
    return _normalize_for_checklist_match(item) in _normalize_for_checklist_match(source_text)


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


def _extract_immutable_candidates(query_text: str) -> List[str]:
    """
    Extract immutable literal candidates from original query_text.
    These are the only types allowed in return_checklist.
    """
    literals: List[str] = []

    # Image placeholders
    for m in re.finditer(r"<image\s*\d+>", query_text, flags=re.IGNORECASE):
        literals.append(m.group(0))

    # LaTeX math blocks (e.g. $NaOH$)
    for m in re.finditer(r"\$[^$]+\$", query_text):
        literals.append(m.group(0))

    # Number + unit patterns
    for m in re.finditer(
        r"\b\d+(?:\.\d+)?\s*(?:mL|L|M|mol|mmol|g|kg|cm|mm|%|K|°C|N)\b",
        query_text,
    ):
        literals.append(m.group(0))

    # Plain numbers (fallback immutable token)
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", query_text):
        literals.append(m.group(0))

    # Chemical formulas without $...$ (e.g. NaOH, H2SO4)
    for m in re.finditer(r"\b(?:[A-Z][a-z]?\d*){2,}\b", query_text):
        literals.append(m.group(0))

    # Acid/base phrases often critical in chemistry QA
    for m in re.finditer(r"\b[a-zA-Z][a-zA-Z\-]*\s+(?:acid|base)\b", query_text):
        literals.append(m.group(0))

    # A few science-critical nouns as literal spans (if present)
    for phrase in ["titration curve", "equivalence point", "pH"]:
        if re.search(re.escape(phrase), query_text, flags=re.IGNORECASE):
            m = re.search(re.escape(phrase), query_text, flags=re.IGNORECASE)
            if m:
                literals.append(query_text[m.start() : m.end()])

    return _dedup_keep_order(literals)


def _auto_extract_immutable_literals(query_text: str) -> List[str]:
    """
    Deterministic fallback when model checklist is unusable.
    Extract high-signal literals that should remain unchanged.
    """
    return _extract_immutable_candidates(query_text)


def sanitize_checklist(return_checklist: Any, query_text: str) -> Tuple[List[str], List[str]]:
    """
    Keep only checklist items that are literal substrings in original query_text.
    Returns (kept_items, dropped_items).
    """
    normalized = _normalize_return_checklist(return_checklist)
    allowed_candidates = _extract_immutable_candidates(query_text)
    allowed_keys = {_normalize_for_checklist_match(x) for x in allowed_candidates}
    kept: List[str] = []
    dropped: List[str] = []
    for item in normalized:
        key = _normalize_for_checklist_match(item)
        if _is_literal_substring(item, query_text) and key in allowed_keys:
            kept.append(item)
        else:
            dropped.append(item)
    return _dedup_keep_order(kept), dropped


def verify_query(
    modified_query_text: str,
    return_checklist: Any,
) -> Tuple[bool, List[str]]:
    """
    Rule-based check: each non-empty checklist entry should still appear in
    `modified_query_text` after normalization.

    Returns:
        (all_passed, missing_items) — missing_items lists checklist strings not found.
    """
    items = _normalize_return_checklist(return_checklist)
    missing: List[str] = []
    for raw in items:
        if not _checklist_item_in_query(modified_query_text, raw):
            missing.append(raw)
    return (len(missing) == 0, missing)


def verify_query_bool(
    modified_query_text: str,
    return_checklist: Any,
) -> bool:
    """Convenience wrapper returning only True/False."""
    ok, _ = verify_query(modified_query_text, return_checklist)
    return ok


def _build_t1_user_text(
    query_text: str,
    options: str,
    target: str,
) -> str:
    action = "T1"
    action_name = "Scientific Proposition & Law Restatement"
    action_prompt = (
        "I plan to paraphrase the scientific question to present the same scientific "
        "proposition/law in a different way. Please assist me in paraphrasing the question, "
        "ensuring that all core keywords, numerical values and scientific concepts are fully "
        "preserved, and the scientific meaning remains completely consistent."
    )
    checklist_block = (
        "Checklist requirement (`return_checklist`):\n"
        " - MUST be a JSON array of literal strings copied from `Original query` only.\n"
        " - Each item must be an exact text span from `Original query` (no paraphrase/synonym/slash-combo).\n"
        " - ONLY include immutable item types: numbers/units, formulas, <image N>, core fixed scientific tokens.\n"
        " - Do NOT include full question sentences or rewriteable wording.\n"
    )
    # " - Bad examples: \"titration curve/graph\", \"best choice/most appropriate indicator\".\n"
    # " - Good examples (must appear exactly): \"100. mL\", \"0.0250 M\", \"acetic acid\", \"$NaOH$\".\n\n"
    return (
        f"Action: {action} - {action_name}\n"
        f"Action instruction:\n{action_prompt}\n\n"
        f"{checklist_block}"
        "Task:\n"
        "1) First decide whether this query text can be modified by this action.\n"
        "2) If it can be modified, provide the modified query.\n"
        "3) If it cannot, explain the reason.\n"
        "4) If can_modify=true, provide `return_checklist` following the strict literal-copy rule; "
        "if can_modify is false, use an empty array [] or briefly state why verification items are N/A.\n\n"
        "Do not modify the options.\n"
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        '  "can_modify": true/false,\n'
        '  "modified_query_text": "string, empty if cannot modify",\n'
        '  "reason": "string",\n'
        '  "return_checklist": "list"\n'
        "}\n\n"
        f"Original query:\n{query_text}\n\n"
        f"Original options:\n{options}\n\n"
        f"Original answer:\n{target}\n\n"
    )


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


def discover_dataset_subject_map(input_path: Path) -> Dict[str, List[str]]:
    """
    Scan input root:
      <input_path>/<dataset>/<subject>_qa.json
    Returns {dataset: [subjects...]}.
    """
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input root dir not found: {input_path}")

    dataset_subjects: Dict[str, List[str]] = {}
    for dataset_dir in sorted([p for p in input_path.iterdir() if p.is_dir()]):
        subjects: List[str] = []
        for file_path in sorted(dataset_dir.glob("*_qa.json")):
            name = file_path.stem  # e.g. chemistry_qa
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


def run_t1_pipeline(
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    client_override: OpenAI | None = None,
) -> Dict[str, Any]:
    """
    Run parse → LLM → parse_output → assemble modified full query → verify.
    Returns `result` (downstream fields) and `trace` (intermediate artifacts).
    """
    query_text, options = parse_query(query)

    user_text = _build_t1_user_text(query_text, options, target)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_text},
    ]

    active_client = client_override or client
    response = active_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )

    model_raw = (response.choices[0].message.content or "").strip()
    model_json = _extract_json_object(model_raw)

    can_modify, modified_query_text, reason, raw_return_checklist = parse_output(model_raw)
    sanitized_checklist, dropped_checklist = sanitize_checklist(raw_return_checklist, query_text)
    if can_modify and not sanitized_checklist:
        sanitized_checklist = _auto_extract_immutable_literals(query_text)

    if can_modify and not modified_query_text:
        can_modify = False
        modified_query_text = query_text
        reason = f"{reason} | Empty modified_query_text from model.".strip(" |")

    if not can_modify:
        modified_query_text = query_text

    sep = "\n" if modified_query_text and options else ""
    modified_full = modified_query_text + sep + options

    verify_ok, verify_missing = verify_query(modified_query_text, sanitized_checklist)

    result = {
        "can_modify": can_modify,
        "original_query": query,
        "original_answer": target,
        "modified_query": modified_full,
        "modified_answer": target,
        "reason": reason,
        "return_checklist": sanitized_checklist,
        "verify_passed": verify_ok,
        "verify_missing_items": verify_missing,
    }

    trace = {
        "parse_query": {
            "query_text": query_text,
            "options": options,
            "query_text_len": len(query_text),
            "options_len": len(options),
        },
        "messages": messages,
        "model_raw": model_raw,
        "model_json": model_json,
        "parse_output": {
            "can_modify": can_modify,
            "modified_query_text": modified_query_text,
            "reason": reason,
            "return_checklist_raw": raw_return_checklist,
            "return_checklist_sanitized": sanitized_checklist,
            "return_checklist_dropped": dropped_checklist,
            "return_checklist_count": len(sanitized_checklist),
        },
        "assemble_modified_query": {
            "separator": sep,
            "modified_full_query": modified_full,
        },
        "verify_inputs": {
            "modified_query_text": modified_query_text,
            "target": target,
            "return_checklist": sanitized_checklist,
        },
        "verify_result": {
            "passed": verify_ok,
            "missing_items": verify_missing,
        },
    }

    return {"result": result, "trace": trace}


def modify_query(
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    out = run_t1_pipeline(query, target, model=model, temperature=temperature)
    return out["result"]


def process_one_item(
    item: Dict[str, Any],
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    include_trace: bool = False,
) -> Dict[str, Any]:
    item_id = item.get("id")
    original_query = str(item.get("query", "") or "")
    original_target = str(item.get("target", "") or "")

    try:
        thread_client = get_thread_client(api_key=api_key, base_url=base_url)
        out = run_t1_pipeline(
            query=original_query,
            target=original_target,
            model=model,
            temperature=temperature,
            client_override=thread_client,
        )
        result = out["result"]
        merged = {"id": item_id, **result}
        if include_trace:
            merged["trace"] = out["trace"]
        return merged
    except Exception as e:
        fallback = {
            "id": item_id,
            "can_modify": False,
            "original_query": original_query,
            "original_answer": original_target,
            "modified_query": original_query,
            "modified_answer": original_target,
            "reason": f"Error: {str(e)}",
            "return_checklist": [],
            "verify_passed": False,
            "verify_missing_items": [],
        }
        if include_trace:
            fallback["trace"] = {"error": str(e)}
        return fallback


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
    include_trace: bool = False,
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
                include_trace,
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
    verify_passed = sum(1 for x in modified_items if x.get("verify_passed"))

    summary = {
        "dataset": dataset,
        "subject": subject,
        "action": action,
        "input_file": str(in_file),
        "output_file": str(out_file),
        "total": len(modified_items),
        "changed": changed,
        "unchanged": unchanged,
        "verify_passed": verify_passed,
        "verify_failed": len(modified_items) - verify_passed,
        "results": modified_items,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _default_example() -> Tuple[str, str]:
    q = (
            "Consider the following drawings below: <image 1> Which of the following statements are true? I: The electrons in each molecule tend to be attracted to the most electronegative element. II: Each molecular drawing follows the localized electron model. III: Both $HF$ and $CO_2$ are linear molecules and therefore nonpolar. IV: The bond angles of $NH_3$ are slightly less than 109.5o because the lone pair compresses the angles between the bonding pairs.\nA. I, III, IV\nB. I, II, IV\nC. I, II, III\nD. II, IV\nE. All the the above statements (I - IV) are true."
    )
    return q, "B·"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="T1 dynamic text (single-case debug or batch dataset processing)."
    )
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
    parser.add_argument(
        "--action",
        default="T1",
        help="Batch mode: action id(s), comma separated. Default T1.",
    )
    parser.add_argument(
        "--input_path",
        default=str(DEFAULT_INPUT_PATH),
        help="Batch mode: input file or input root dir.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Batch mode: output file or output dir.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Full question string (stem + options). Ignored if --query-file is set.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="UTF-8 text file containing the full question.",
    )
    parser.add_argument(
        "--target",
        "-a",
        type=str,
        default=None,
        help="Ground-truth answer (e.g. option letter).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--include_trace",
        action="store_true",
        help="Batch mode: include trace for each sample in output JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write full JSON (result + trace) to this path.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run built-in chemistry MCQ example (sets query + target).",
    )
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
                    include_trace=args.include_trace,
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

    if args.example:
        query, target = _default_example()
    else:
        if args.query_file is not None:
            query = args.query_file.read_text(encoding="utf-8")
        elif args.query is not None:
            query = args.query
        else:
            parser.error("Provide --query / --query-file, or use --example.")
        if args.target is None:
            parser.error("--target / -a is required unless --example is set.")
        target = args.target

    out = run_t1_pipeline(
        query,
        target,
        model=args.model,
        temperature=args.temperature,
    )

    sections = {
        "1_parse_query_options": out["trace"]["parse_query"],
        "2_model_raw": out["trace"]["model_raw"],
        "3_model_json_parsed": out["trace"]["model_json"],
        "4_parse_output_checklist": out["trace"]["parse_output"],
        "5_verify_inputs": out["trace"]["verify_inputs"],
        "6_verify_result": out["trace"]["verify_result"],
        "7_final_result": out["result"],
    }

    for title, payload in sections.items():
        print(f"\n{'=' * 20} {title} {'=' * 20}\n")
        if title == "2_model_raw":
            print(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload_out = {"result": out["result"], "trace": out["trace"]}
        args.output.write_text(
            json.dumps(payload_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote full bundle to {args.output}")


if __name__ == "__main__":
    main()
