
"""HR-Bench (high-resolution perception) task helpers for lmms_eval — reads the EXISTING parquet.

HR-Bench (Wang et al., https://github.com/DreamMr/HR-Bench) is a 4K / 8K high-resolution
multiple-choice benchmark with a per-question ``category``. Scored on multiple-choice accuracy,
overall + per category. (Per-row accuracy = the official ``acc``; ``acc+`` cyclic-consistency is
not computed.)

The yaml loads the HF repo's ``hr_bench_{4k,8k}.parquet`` as is (a one-file 'train' split, columns
``index, answer, question, A, B, C, D, category, image``), where ``image`` is a BASE64 JPEG string
and the options live in separate columns.
``doc_to_visual`` decodes the base64 image; ``doc_to_text`` folds the options into the question.

The Imagine model reasons with abstract visual tokens then emits ``\\boxed{letter}``;
``hrbench_process_results`` reads the boxed answer back.
"""

from __future__ import annotations

import base64
import io
import re

from lmms_eval.tasks._task_utils.eval_utils import (
    extract_final_boxed_content,
)
from loguru import logger as eval_logger
from PIL import Image

_OPTIONS = ("A", "B", "C", "D")


def hrbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Decode the row's base64 JPEG ``image`` -> [PIL]."""
    raw = doc["image"]
    if isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:  # base64 string (optionally a data: URI)
        text = str(raw).split("base64,", 1)[-1]
        data = base64.b64decode(text)
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def hrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Question + inline ``(A)..(D)`` options (from the separate columns), then the post-prompt
    (the \\boxed instruction — set in the yaml so training + eval share the answer format)."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    opts = "\n".join(f"({L}) {doc[L]}" for L in _OPTIONS if doc.get(L) is not None)
    question = f"{str(doc.get('question', '')).strip()}\n{opts}"
    return "\n".join(part for part in (pre, question, post) if part)


def _gt_letter(doc: dict) -> str:
    answer = str(doc.get("answer", "")).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    m = re.match(r"\(?([A-H])[.):]?\b", answer, re.IGNORECASE)
    return m[1].upper() if m else "?"


def hrbench_doc_to_target(doc):
    return _gt_letter(doc)


def _extract_letter(text: str, valid_letters) -> str:
    """Recover the predicted option letter: prefer an answer marker, else a bare letter."""
    cleaned = text or ""
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    letters = "".join(dict.fromkeys(c.upper() for c in valid_letters)) or "ABCD"
    marker = re.compile(
        rf"(?:answer|option|choice|conclusion)\b\s*(?:is|are|:|=|->|-|—)?\s*"
        rf"\(?\*{{0,2}}([{letters}])\*{{0,2}}\)?\b",
        re.IGNORECASE,
    )
    found = marker.findall(cleaned)
    if found:
        return found[-1].upper()
    bare = re.findall(
        rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", cleaned, re.IGNORECASE
    )
    return bare[0].upper() if bare else ""


def _parse_options(doc: dict) -> dict[str, str]:
    """Option letter -> text, from the separate A/B/C/D columns."""
    return {L: str(doc[L]) for L in _OPTIONS if doc.get(L) is not None}


def _resolve_letter(boxed: str, options: dict[str, str]) -> str:
    """Map the boxed answer to an option letter. Imagine (trained on descriptive boxed answers) often
    boxes the option's TEXT (``\\boxed{a wheelchair}``) rather than its letter, so match the boxed
    content against the option texts: exact letter -> exact text -> containment -> a letter marker."""
    b = (boxed or "").strip()
    if not b:
        return ""
    m = re.match(r"^\(?\s*([A-Za-z])\s*\)?[.):]?\s*$", b)  # bare / (A) / A.
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    bl = b.lower().strip().rstrip(".").strip()
    for letter, opt in options.items():  # exact option-text match
        if bl == opt.lower():
            return letter
    for letter, opt in options.items():  # containment either direction
        o = opt.lower()
        if o and (o in bl or bl in o):
            return letter
    return _extract_letter(b, list(options) or list("ABCD"))


def hrbench_process_results(doc, results):
    """Read the model's ``\\boxed{}`` answer, map to an option letter, compare to gold."""
    pred = results[0] if results else ""
    boxed = extract_final_boxed_content(pred, strict=False)
    gt_letter = _gt_letter(doc)
    pred_letter = _resolve_letter(boxed, _parse_options(doc))
    return {
        "accuracy": {
            "category": str(doc.get("category") or "overall"),
            "score": float(pred_letter == gt_letter and pred_letter != ""),
            "pred": pred_letter,
            "gt": gt_letter,
        }
    }


def hrbench_aggregate_accuracy(results, args=None):
    if not results:
        return 0.0
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "overall"), []).append(r["score"])
    for cat, scores in sorted(by_cat.items()):
        eval_logger.info(
            f"HR-Bench [{cat}]: {sum(scores) / len(scores) * 100:.2f}% (n={len(scores)})"
        )
    return float(sum(r["score"] for r in results) / len(results) * 100.0)
