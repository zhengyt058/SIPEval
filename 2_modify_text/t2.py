#!/usr/bin/env python3
"""
科学等价改写流水线（占位符 + 库校验）

步骤概览：
1) 模型门控：仅决定 can_modify（能否走本流水线）；否：仅填 reason，其余列表类字段为空
2) can_modify 为 true：抽取 items → 改写 → 校验；reason 恒为空（reason 只表示「不能改」的原因）
3) verify_passed：can_modify 为 false 时为 null（未做校验）；can_modify 为 true 时为是否至少一处校验通过
4) 校验全失败时重试 transform 最多 3 次；仍失败则 verify_passed=false，modified_query 保持原文

批量：与 t1 相同约定 —— original_qa/<dataset>/qa.json，输出 modified_qa/<dataset>_T2_qa.json（或 --output_path 为具体 .json）。
示例：python t3.py --dataset MMMU-Chemistry --action T2 --input_path ... --output_path ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# 与 t1.py 对齐的默认推理服务（可通过环境变量覆盖）
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-mYDVzMMHpaUyuSWqwi42JaUs1ZNCBKaYxQuTHNX1siu2wEVG")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
TIMEOUT = 120
MAX_TOKENS = 2048
TEMPERATURE = 0.0
# 全部校验失败时，重新调用 transform 的最大次数（不含首次）
MAX_TRANSFORM_RETRY = 3
DEFAULT_MAX_WORKERS = 8
DEFAULT_INPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data")
DEFAULT_OUTPUT_PATH = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_data")

client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
_thread_local = threading.local()


def get_thread_client(api_key: str, base_url: str) -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            timeout=TIMEOUT,
        )
    return _thread_local.client


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


Kind = Literal["unit", "math", "chem"]


@dataclass
class EquivItem:
    kind: Kind
    original: str
    placeholder: str


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
    model: str = DEFAULT_MODEL,
    *,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> dict:
    active = client_override if client_override is not None else client
    response = active.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _extract_json_object(raw)


# ---------------------------------------------------------------------------
# 0. 门控：是否适合本流水线改写
# ---------------------------------------------------------------------------

SYSTEM_GATE = """You are an assistant for science-question rewriting strategy. Output valid JSON only, no markdown.
Task: Determine whether the given question stem is **suitable** for downstream "scientific-equivalent fragment replacement" (e.g., unit conversion, mathematically equivalent transformation, chemical formula/SMILES equivalence), while still allowing automatic verification via rules such as SymPy, pint, and RDKit.

If the question is purely descriptive with no numbers/no separable fragments, or rewriting would inevitably damage the meaning or answer uniqueness, or it is not suitable for automatic verification, set can_modify = false.
Otherwise set can_modify = true, and provide a brief reason describing what can be rewritten."""


def judge_can_modify(
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    *,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> Tuple[bool, str]:
    user = (
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<ANSWER>>>\n"
        f"{target}\n"
        "<<<ANSWER_END>>>\n\n"
        'Output JSON only: {"can_modify": true/false, "reason": "brief explanation in English"}'
    )
    obj = _chat_json(
        SYSTEM_GATE, user, model=model, client_override=client_override, temperature=temperature
    )
    cm = bool(obj.get("can_modify", False))
    reason = str(obj.get("reason", "") or "").strip()
    return cm, reason


# ---------------------------------------------------------------------------
# 1–2. 识别可做科学等价的片段，并生成带占位符的 query
# ---------------------------------------------------------------------------

SYSTEM_IDENTIFY = """You are an assistant for analyzing science questions. Output valid JSON only, no markdown.
Task: Find contiguous text spans in the question stem that can be rewritten into scientifically equivalent forms.

[Scientific equivalence - unified definition]
Two expressions are equivalent only if they refer to the same scientific object or the same numerical relationship before and after rewriting. Ideally, the rewrite should be verifiable by rules from SymPy (math), pint (units), or RDKit/InChI (chemical structure).

[Meaning of each kind]
- unit (numbers and units): Different valid expressions of the same physical quantity. Example: 2801 kJ vs 2.801×10^6 J. SI prefixes or scientific notation are allowed if meaning is unchanged. If the question has strict constraints on significant figures or formatting, do not choose spans that would break those constraints.
- math (mathematical expression): Expressions that are mathematically identical under the same variable conventions. Example: $a+b=c$ vs $b+a=c$. Equivalent transformation must not change the domain or stoichiometric relationships used in the question.
- chem (chemical structure/formula): **Scientific equivalence includes but is not limited to** (1) **interchange between a molecular formula (or chemical shorthand in the question) and a SMILES representing the same structure**; (2) **different SMILES notations that correspond to the same molecular graph (same compound / same InChI-represented structure)**; 
Rules:
- Each span must be a contiguous substring from the original question text.
- Use placeholders {{PH0}}, {{PH1}}, ... in the order spans appear in the question.
- Do not replace literals like <image N>, or literals required by explicit constraints such as "write only the number", unless that specific span truly allows an equivalent rewrite.

[Output constraint]
- Output only the JSON field items (array).
"""


def _normalize_placeholder(placeholder: str, index: int) -> str:
    """统一为 {{PH0}} 形式。"""
    p = (placeholder or "").strip()
    m = re.fullmatch(r"\{\{PH(\d+)\}\}", p)
    if m:
        return f"{{{{PH{m.group(1)}}}}}"
    m = re.fullmatch(r"\{PH(\d+)\}", p)
    if m:
        return f"{{{{PH{m.group(1)}}}}}"
    return f"{{{{PH{index}}}}}"


def build_templated_query(query: str, items: List[EquivItem]) -> str:
    """用题干 + items 确定性生成带占位符的题干，与模型输出无关。"""
    out = query
    for it in items:
        if it.original not in out:
            continue
        out = out.replace(it.original, it.placeholder, 1)
    return out


def identify_equivalences(
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    *,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> Tuple[List[EquivItem], str]:
    user = (
        "The two blocks <<<QUERY>>> and <<<ANSWER>>> are provided only for understanding the problem. Your JSON must **not** include these tags, must not repeat this instruction, and must not include words like schema or Gold answer.\n\n"
        "<<<QUERY>>>\n"
        f"{query}\n"
        "<<<QUERY_END>>>\n\n"
        "<<<ANSWER>>>\n"
        f"{target}\n"
        "<<<ANSWER_END>>>\n\n"
        "Return exactly one JSON object with **only the key items** (do not output templated_query):\n"
        '{"items": [ {"kind": "unit|math|chem", "original": "must be a contiguous substring from the question", "placeholder": "{{PH0}}"}, ... ] }\n\n'
        "Rules: list items from left to right according to span order in the question; each original must truly appear between <<<QUERY>>> and <<<QUERY_END>>>; "
        "placeholders must be {{PH0}}, {{PH1}}, ... in order. If no rewriteable span exists, return only: "
        '{"items": []}'
    )
    obj = _chat_json(
        SYSTEM_IDENTIFY, user, model=model, client_override=client_override, temperature=temperature
    )
    items_raw = obj.get("items") or []

    items: List[EquivItem] = []
    for i, row in enumerate(items_raw):
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", "math")).lower()
        if kind not in ("unit", "math", "chem"):
            kind = "math"
        original = str(row.get("original", "")).strip()
        if not original or original not in query:
            continue
        ph = _normalize_placeholder(str(row.get("placeholder", "")), i)
        items.append(EquivItem(kind=kind, original=original, placeholder=ph))

    templated = query if not items else build_templated_query(query, items)
    return items, templated


# ---------------------------------------------------------------------------
# 3. 对每个片段生成等价改写
# ---------------------------------------------------------------------------

SYSTEM_TRANSFORM = """You are an assistant for rewriting scientific expressions. Output valid JSON only.
For each given original span, produce one scientifically equivalent transformed span: make the surface form as different as possible while strictly satisfying the equivalence definition below.

[Scientific equivalence]
- unit: Represents the same physical quantity as original; unit conversion or normalized notation is allowed, with unchanged physical meaning.
- math: Mathematically identical to original under the same symbol conventions.
- chem: **Scientific equivalence includes** (1) **interchange between molecular formula (or chemical shorthand in the question) and a SMILES representing the same structure**; (2) **different SMILES notations are equivalent if they represent the same molecular structure (same connectivity / same compound)**; (3) do not replace with a different compound or different isomer (unless isomerism is explicitly discussed and the span allows it).

Do not add explanations; output JSON only."""


def transform_equivalences(
    items: List[EquivItem],
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    *,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> List[str]:
    if not items:
        return []
    payload = [
        {"kind": it.kind, "original": it.original, "placeholder": it.placeholder}
        for it in items
    ]
    user = f"""Context query:
{query}

Answer:
{target}

Items to transform (same order):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON:
{{ "transformed": [ "string for item 0", "string for item 1", ... ] }}
Length of transformed MUST equal {len(items)}."""
    obj = _chat_json(
        SYSTEM_TRANSFORM, user, model=model, client_override=client_override, temperature=temperature
    )
    arr = obj.get("transformed")
    if not isinstance(arr, list):
        arr = []
    out = [str(x).strip() for x in arr]
    while len(out) < len(items):
        out.append(items[len(out)].original)
    return out[: len(items)]


# ---------------------------------------------------------------------------
# 4. 库校验：SymPy / pint / RDKit
# ---------------------------------------------------------------------------

def _strip_latex_dollars(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    return s


def _latexish_to_sympy_str(s: str) -> str:
    s = _strip_latex_dollars(s)
    s = s.replace(" ", "")
    s = s.replace("\\rightarrow", "-").replace("→", "-")
    s = re.sub(r"\\Delta", "Delta", s)
    s = re.sub(r"\\degree", "", s)
    s = re.sub(r"\^\\circ", "", s)
    # 简单幂：x^{2} -> x**2
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^(\d)", r"**\1", s)
    s = s.replace("^", "**")
    return s


def math_equivalent(a: str, b: str) -> bool:
    try:
        import sympy as sp

        sa = _latexish_to_sympy_str(a)
        sb = _latexish_to_sympy_str(b)
        ea = sp.sympify(sa)
        eb = sp.sympify(sb)
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


def _normalize_unit_text(s: str) -> str:
    s = s.replace(",", "").strip()
    return s


def unit_equivalent(a: str, b: str) -> bool:
    try:
        from pint import UnitRegistry

        ureg = UnitRegistry()
        qa = ureg(_normalize_unit_text(a))
        qb = ureg(_normalize_unit_text(b))
        if qa.dimensionality != qb.dimensionality:
            return False
        mag_a = qa.to(qb.units).magnitude
        mag_b = qb.magnitude
        scale = max(abs(mag_b), abs(mag_a), 1.0)
        return abs(mag_a - mag_b) <= 1e-9 * scale
    except Exception:
        return False


def _strip_latex_chem_formula(s: str) -> str:
    """LaTeX 下标 → 普通分子式，如 C_6H_{12}O_6、CO_2 → C6H12O6、CO2。"""
    s = (s or "").strip().strip("$")
    s = re.sub(r"\s+", "", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"([A-Z][a-z]?)\_\{(\d+)\}", r"\1\2", s)
        s = re.sub(r"([A-Z][a-z]?)\_(\d)", r"\1\2", s)
    return s


# 行末相态：气/液/固/水溶液等（题干与反应式中常见）
_CHEM_PHASE_SUFFIX = re.compile(r"\s*\((?:aq|s|l|g|cr|amorph)\)\s*$", re.IGNORECASE)


def _preprocess_chem_species(raw: str) -> str:
    """
    化学片段预处理，便于比对「6 $O_2$(g)」与「6 O=O(g)」等：
    - 去掉行首化学计量数（如 6、12）
    - 去掉 $ 定界符
    - LaTeX 下标展开（沿用 _strip_latex_chem_formula）
    - 去掉行末相态 (g)/(l)/(s)/(aq) 等
    - 去掉空白
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"^\d+\s*", "", s)
    s = s.replace("$", "")
    s = _strip_latex_chem_formula(s)
    s = _CHEM_PHASE_SUFFIX.sub("", s.strip())
    s = re.sub(r"\s+", "", s)
    return s


def _parse_hill_formula_counts(formula: str) -> Optional[Dict[str, int]]:
    """将 Hill 风格分子式（如 C6H12O6、CO2）解析为元素计数字典。"""
    formula = re.sub(r"\s+", "", (formula or "").strip())
    if not formula:
        return None
    counts: Dict[str, int] = {}
    i = 0
    n = len(formula)
    while i < n:
        if not formula[i].isupper():
            return None
        el = formula[i]
        i += 1
        if i < n and formula[i].islower():
            el += formula[i]
            i += 1
        start = i
        while i < n and formula[i].isdigit():
            i += 1
        num_str = formula[start:i]
        cnt = int(num_str) if num_str else 1
        counts[el] = counts.get(el, 0) + cnt
    return counts


def _mol_formula_counts(mol: Any) -> Optional[Dict[str, int]]:
    try:
        from rdkit.Chem import rdMolDescriptors

        if mol is None:
            return None
        f = rdMolDescriptors.CalcMolFormula(mol)
        return _parse_hill_formula_counts(f)
    except Exception:
        return None


def _silence_rdkit_parse_errors() -> None:
    """MolFromSmiles 对非 SMILES（如 LaTeX 分子式）会失败，属预期；关闭 RDKit 刷屏到 stderr。"""
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.error")
    except Exception:
        pass


def chem_equivalent(a: str, b: str) -> bool:
    """
    化学等价：
    - 先经 _preprocess_chem_species（计量数、$、相态等）再比对。
    - 两侧均为合法 SMILES：同一 InChI（同一结构）。
    - 一侧为 LaTeX/文本分子式、一侧为 SMILES：元素组成一致（经验式一致）则通过
      （如 CO_2 与 O=C=O；C_6H_{12}O_6 与某葡萄糖 SMILES）。
    - 两侧均非 SMILES：剥离 LaTeX 后比较元素计数。
    """
    sa0, sb0 = a.strip(), b.strip()
    if sa0 == sb0:
        return True
    sa = _preprocess_chem_species(a)
    sb = _preprocess_chem_species(b)
    if sa == sb:
        return True
    if not sa and not sb:
        return True
    if not sa or not sb:
        return False
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi
    except ImportError:
        return sa == sb

    _silence_rdkit_parse_errors()
    ma = Chem.MolFromSmiles(sa)
    mb = Chem.MolFromSmiles(sb)

    if ma is not None and mb is not None:
        try:
            return inchi.MolToInchi(ma) == inchi.MolToInchi(mb)
        except Exception:
            return False

    # 一侧 SMILES、一侧分子式（含 LaTeX）
    if ma is not None and mb is None:
        fc = _mol_formula_counts(ma)
        tc = _parse_hill_formula_counts(_strip_latex_chem_formula(sb))
        return fc is not None and tc is not None and fc == tc
    if mb is not None and ma is None:
        fc = _mol_formula_counts(mb)
        tc = _parse_hill_formula_counts(_strip_latex_chem_formula(sa))
        return fc is not None and tc is not None and fc == tc

    # 纯分子式对分子式
    ca = _parse_hill_formula_counts(_strip_latex_chem_formula(sa))
    cb = _parse_hill_formula_counts(_strip_latex_chem_formula(sb))
    if ca is not None and cb is not None:
        return ca == cb

    return sa == sb


def pair_equivalent(kind: Kind, original: str, transformed: str) -> bool:
    if original.strip() == transformed.strip():
        return True
    if kind == "math":
        return math_equivalent(original, transformed)
    if kind == "unit":
        return unit_equivalent(original, transformed)
    if kind == "chem":
        return chem_equivalent(original, transformed)
    return False


def verify_all(
    items: List[EquivItem], transformed: List[str]
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """
    Returns:
        all_ok — 全部通过为 True
        details — 每一项一条：kind, original, transformed, passed
        failure_reasons — 仅未通过项的说明（便于快速扫错）
    """
    details: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    for it, new in zip(items, transformed):
        passed = pair_equivalent(it.kind, it.original, new)
        details.append(
            {
                "kind": it.kind,
                "original": it.original,
                "transformed": new,
                "passed": passed,
            }
        )
        if not passed:
            failure_reasons.append(f"fail {it.kind}: {it.original!r} vs {new!r}")
    all_ok = len(failure_reasons) == 0
    return all_ok, details, failure_reasons


# ---------------------------------------------------------------------------
# 5–6. 回填占位符 → modified_query
# ---------------------------------------------------------------------------

def apply_placeholders(templated_query: str, items: List[EquivItem], replacements: List[str]) -> str:
    out = templated_query
    for it, new in zip(items, replacements):
        out = out.replace(it.placeholder, new, 1)
    return out


def run_pipeline(
    query: str,
    target: str,
    model: str = DEFAULT_MODEL,
    record_id: str | int | None = None,
    *,
    client_override: OpenAI | None = None,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """端到端：门控 → 抽取 → 改写与校验（失败则最多重试 MAX_TRANSFORM_RETRY 次）。

    can_modify 仅表示门控「本题是否适合走本流水线」，不因后续抽取/校验结果而改变。
    reason 仅在 can_modify 为 false 时有效，为模型给出的「不能改」说明；为 true 时恒为空串。
    verify_passed：门控未通过时为 None；门控通过后为是否至少一处库校验通过。
    """
    original_query = query
    original_answer = target

    gate_can, gate_reason = judge_can_modify(
        query, target, model=model, client_override=client_override, temperature=temperature
    )

    if not gate_can:
        return {
            "id": record_id,
            "can_modify": False,
            "original_query": original_query,
            "original_answer": original_answer,
            "modified_query": original_query,
            "modified_answer": original_answer,
            "reason": gate_reason or "The model judged this question as unsuitable for rewriting.",
            "return_checklist": [],
            "verify_passed": None,
            "verify_error_items": [],
            "retry_times": 0,
        }

    items, templated = identify_equivalences(
        query, target, model=model, client_override=client_override, temperature=temperature
    )

    if not items:
        return {
            "id": record_id,
            "can_modify": True,
            "original_query": original_query,
            "original_answer": original_answer,
            "modified_query": original_query,
            "modified_answer": original_answer,
            "reason": "",
            "return_checklist": [],
            "verify_passed": False,
            "verify_error_items": [],
            "retry_times": 0,
        }

    retry_times = 0
    transformed = transform_equivalences(
        items, query, target, model=model, client_override=client_override, temperature=temperature
    )
    _, verify_details, _ = verify_all(items, transformed)

    while (
        verify_details
        and all(not row["passed"] for row in verify_details)
        and retry_times < MAX_TRANSFORM_RETRY
    ):
        retry_times += 1
        transformed = transform_equivalences(
            items, query, target, model=model, client_override=client_override, temperature=temperature
        )
        _, verify_details, _ = verify_all(items, transformed)

    any_pass = any(row["passed"] for row in verify_details)

    return_checklist = [
        {
            "kind": row["kind"],
            "original": row["original"],
            "modified": row["transformed"] if row["passed"] else row["original"],
            "model_transformed": row["transformed"],
            "passed": row["passed"],
        }
        for row in verify_details
    ]

    verify_error_items = [
        {
            "kind": row["kind"],
            "original": row["original"],
            "transformed": row["transformed"],
            "reason": "Library verification failed (not equivalent under SymPy/pint/RDKit rules).",
        }
        for row in verify_details
        if not row["passed"]
    ]

    if not any_pass:
        return {
            "id": record_id,
            "can_modify": True,
            "original_query": original_query,
            "original_answer": original_answer,
            "modified_query": original_query,
            "modified_answer": original_answer,
            "reason": "",
            "return_checklist": return_checklist,
            "verify_passed": False,
            "verify_error_items": verify_error_items,
            "retry_times": retry_times,
        }

    replacements = [
        row["transformed"] if row["passed"] else row["original"] for row in verify_details
    ]
    modified_query = apply_placeholders(templated, items, replacements)

    return {
        "id": record_id,
        "can_modify": True,
        "original_query": original_query,
        "original_answer": original_answer,
        "modified_query": modified_query,
        "modified_answer": original_answer,
        "reason": "",
        "return_checklist": return_checklist,
        "verify_passed": True,
        "verify_error_items": verify_error_items,
        "retry_times": retry_times,
    }


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
        return run_pipeline(
            query=original_query,
            target=original_target,
            model=model,
            record_id=item_id,
            client_override=thread_client,
            temperature=temperature,
        )
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
            "verify_passed": None,
            "verify_error_items": [],
            "retry_times": 0,
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
    verify_passed_n = sum(1 for x in modified_items if x.get("verify_passed") is True)

    summary = {
        "dataset": dataset,
        "subject": subject,
        "action": action,
        "input_file": str(in_file),
        "output_file": str(out_file),
        "total": len(modified_items),
        "changed": changed,
        "unchanged": unchanged,
        "verify_passed": verify_passed_n,
        "verify_failed": len(modified_items) - verify_passed_n,
        "results": modified_items,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T2 scientific-equivalent rewriting: single-case debug or batch processing by dataset (same directory/argument conventions as t1)."
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
        default="T2",
        help="Batch mode: action identifiers, comma-separated. Default: T2.",
    )
    parser.add_argument(
        "--input_path",
        default=str(DEFAULT_INPUT_PATH),
        help="Batch mode: input qa.json file or root directory of original_qa.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Batch mode: output JSON file or modified_qa directory.",
    )
    parser.add_argument("--query", type=str, default=None, help="Single-case mode: full question stem")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Single-case mode: UTF-8 file containing the question stem",
    )
    parser.add_argument("--target", "-a", type=str, default=None, help="Single-case mode: gold answer")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Single-case mode: write JSON result to this path",
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
        all_summaries: List[Dict[str, Any]] = []
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
                    f"unchanged={summary['unchanged']}, verify_passed={summary['verify_passed']}, "
                    f"verify_failed={summary['verify_failed']}\n"
                    f"output={summary['output_file']}"
                )

        grand_total = sum(x["total"] for x in all_summaries)
        grand_changed = sum(x["changed"] for x in all_summaries)
        grand_verify = sum(x["verify_passed"] for x in all_summaries)
        print(
            f"\nAll batch tasks finished. files={len(all_summaries)}, total={grand_total}, "
            f"changed={grand_changed}, verify_passed_items={grand_verify}"
        )
        return

    if args.query_file is not None:
        query = args.query_file.read_text(encoding="utf-8")
    elif args.query is not None:
        query = args.query
    else:
        query = (
            "The graph below shows the titration curve that results when 100. mL of 0.0250 M acetic acid "
            "is titrated with 0.100 M $NaOH$. <image 1> Which of the following indicators is the best choice "
            "for this titration?\nA. Methyl orange, whose pH range of color change is 3.2 - 4.4 .\n"
            "B. Methyl red, whose pH range of color change is 4.8 - 6.0 .\n"
            "C. MBromothymol blue, whose pH range of color change is 6.1 - 7.6 .\n"
            "D. Phenolphthalein, whose pH range of color change is 8.2 - 10.0 .\n"
            "E. Alizarin, whose pH range of color change is 11.0 - 12.4 ."
        )

    if args.query_file is not None or args.query is not None:
        if args.target is None:
            parser.error("--target / -a is required when using --query or --query-file.")

    target = args.target if args.target is not None else "D"

    result = run_pipeline(
        query,
        target,
        model=args.model,
        record_id="demo",
        temperature=args.temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nWrote to {args.output}")


if __name__ == "__main__":
    main()
