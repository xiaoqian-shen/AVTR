
"""V*Bench (visual search) task helpers for lmms_eval.

V*Bench (Wu & Xie, https://github.com/penghao-wu/vstar) is a single-image multiple-choice
benchmark for fine-grained visual search (categories: ``direct_attributes``, ``relative_position``).
Scored on multiple-choice accuracy, overall + per category.

The yaml loads the HF repo's ``test_questions.jsonl`` (a one-file 'train' split); each row is
``{image, text, category, question_id, label}`` where ``image`` is a path RELATIVE to the repo
root (e.g. ``direct_attributes/sa_4690.jpg``). ``doc_to_visual`` fetches that file from the same
repo on demand (cached by huggingface_hub).

The Imagine model reasons with abstract visual tokens then emits ``\\boxed{letter}``;
``vstar_process_results`` reads the boxed answer back.
"""

from __future__ import annotations

import re

from lmms_eval.tasks._task_utils.eval_utils import (
    extract_final_boxed_content,
)
from loguru import logger as eval_logger
from PIL import Image

# The HF dataset repo the yaml loads; doc["image"] is a file path inside it.
VSTAR_REPO = "craigwu/vstar_bench"
# "(A) rubber" / "A. rubber" -> (letter, option text).
_RE_OPTION = re.compile(r"(?m)^\s*\(?([A-H])[.):]\s+(.*\S)\s*$")
_RE_ANSWER_INSTR = re.compile(r"^\s*answer with\b.*$", re.IGNORECASE | re.MULTILINE)


def vstar_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Fetch the row's image (a file of VSTAR_REPO) -> [PIL]."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(VSTAR_REPO, doc["image"], repo_type="dataset")
    return [Image.open(local).convert("RGB")]


def vstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Question + inline options (V*Bench's own 'answer directly' trailer stripped), then the
    post-prompt (the \\boxed instruction — set in the yaml so training + eval share the format)."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    question = _RE_ANSWER_INSTR.sub("", str(doc.get("text", ""))).strip()
    return "\n".join(part for part in (pre, question, post) if part)


def _parse_options(doc: dict) -> dict[str, str]:
    """Parse the inline ``(A) rubber`` option lines out of the question -> {letter: text}."""
    return {
        m.group(1).upper(): m.group(2).strip()
        for m in _RE_OPTION.finditer(str(doc.get("text", "")))
    }


def _gt_letter(doc: dict) -> str:
    answer = str(doc.get("label", "")).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    m = re.match(r"\(?([A-H])[.):]?\b", answer, re.IGNORECASE)
    return m[1].upper() if m else "?"


def vstar_doc_to_target(doc):
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


def _resolve_letter(boxed: str, options: dict[str, str]) -> str:
    """Map the boxed answer to an option letter. Imagine (trained on descriptive boxed answers) often
    boxes the option's TEXT (``\\boxed{rubber}``) rather than its letter, so match the boxed content
    against the option texts: exact letter -> exact text -> containment -> a letter marker."""
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


def vstar_process_results(doc, results):
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


def vstar_aggregate_accuracy(results, args=None):
    """Overall accuracy (%); logs the per-category breakdown."""
    if not results:
        return 0.0
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "overall"), []).append(r["score"])
    for cat, scores in sorted(by_cat.items()):
        eval_logger.info(
            f"V*Bench [{cat}]: {sum(scores) / len(scores) * 100:.2f}% (n={len(scores)})"
        )
    return float(sum(r["score"] for r in results) / len(results) * 100.0)
