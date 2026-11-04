"""BLINK (multi-image visual perception) task helpers for lmms_eval.

BLINK (Fu et al., https://huggingface.co/datasets/BLINK-Benchmark/BLINK) is a
multiple-choice benchmark across perception sub-tasks (Art Style, Counting, Relative
Depth, Spatial Relation, Visual Correspondence, ...), each question grounded in one OR
MORE images. We evaluate the ``val`` split (``test`` has no public answers). Scored on a
single axis -- multiple-choice answer accuracy -- reported overall and per ``sub_task``.

Each ``doc`` is a row of the official per-sub-task ``val`` parquets: ``idx``, ``prompt`` (BLINK's
preformatted prompt with the lettered choices -- what the model is asked; the bare ``question``
is not used), ``answer`` (``"(B)"``), ``sub_task`` and ``image_1``..``image_4`` (unused slots are
null). Rows with ``question`` / ``images`` columns in place of ``prompt`` / ``image_k`` (a combined
repack of the same split) are read the same way. The yaml wires the functions below via
``!function utils.<name>``.
"""

from __future__ import annotations

import ast
import base64
import io
import re

from loguru import logger as eval_logger
from PIL import Image

# An option label at the start of a line: "A. ", "B) ", "(C) " (optionally parenthesized).
_OPTION_LABEL_RE = re.compile(r"(?m)^\s*\(?([A-H])[.):]\s+")


def _to_pil(raw: object) -> Image.Image:
    """Decode one raw image element (bytes / base64 / path / PIL / HF struct) to RGB PIL."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):  # HF Image feature: {"bytes": ..., "path": ...}
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(str(raw["path"])).convert("RGB")
        raise ValueError(f"Image dict has neither bytes nor path: {list(raw)}")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if isinstance(raw, str):
        text = raw.split("base64,", 1)[-1] if raw.startswith("data:image") else raw
        try:
            return Image.open(io.BytesIO(base64.b64decode(text))).convert("RGB")
        except Exception:  # noqa: BLE001 -- not base64; fall back to a file path
            return Image.open(raw).convert("RGB")
    raise ValueError(f"Unsupported image element of type {type(raw).__name__}")


def _prompt(doc: dict) -> str:
    """The text the model is asked: upstream ``prompt``, or ``question`` in a repack that stores it there."""
    value = doc.get("prompt")
    return str(value if value is not None else doc.get("question", ""))


def _images_field(doc: dict) -> list:
    """Return the per-doc images as a list: upstream ``image_1``..``image_4`` in order without
    the null slots, or an ``images`` / ``image`` column (parsing a string-encoded list if needed)."""
    value = doc.get("images")
    if value is None:
        value = doc.get("image")
    if value is None:
        return [doc[k] for k in ("image_1", "image_2", "image_3", "image_4") if doc.get(k) is not None]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def blink_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Return the question's images as a list of PIL.Image (the model base64-encodes them).

    Every BLINK question has at least one image, so a row without any means the data lacks the
    image columns; evaluating it text-only would still produce a normal-looking, near-chance score.
    """
    images = _images_field(doc)
    if not images:
        raise ValueError(
            f"BLINK row {doc.get('idx', doc.get('id'))!r} has no images: expected image_1..image_4 "
            "(or an `images` column; see lmms_eval/tasks/blink/utils.py)"
        )
    return [_to_pil(img) for img in images]


def blink_process_docs(dataset):
    """Cast the raw ``list<binary>`` images column to an HF ``Image`` feature.

    Run at load time (yaml ``process_docs``) so (a) ``doc_to_visual`` gets decoded PIL
    images and (b) lmms_eval drops them from the ``--log_samples`` copy -- its
    ``dataset_no_image`` only strips ``Image``/``Audio`` features, so a plain
    ``list<binary>`` column would otherwise be stringified into huge sample logs.
    No-op if the column is already an ``Image`` feature or absent.
    """
    import datasets

    feature = getattr(dataset, "features", {}).get("images")
    already_image = isinstance(feature, datasets.Sequence) and isinstance(
        feature.feature, datasets.Image
    )
    if feature is not None and not already_image:
        dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    return dataset


def blink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Render the prompt text: optional pre-prompt, the question (choices inline), post-prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = _prompt(doc).strip()
    return "\n".join(part for part in (pre_prompt, question, post_prompt) if part)


def _option_letters(doc: dict) -> list[str]:
    """Valid option letters: scanned from inline labels, else A-D (no BLINK question has more options)."""
    found = list(dict.fromkeys(_OPTION_LABEL_RE.findall(_prompt(doc))))
    return found or list("ABCD")


def _gt_letter(doc: dict) -> str:
    """Map the ground-truth answer to its option letter (A-H), or ``"?"``.

    Upstream BLINK stores ``answer`` as ``"(B)"``; both that and a bare ``"B"`` map to
    ``"B"``.
    """
    answer = str(doc.get("answer", "")).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    match = re.match(r"\(?([A-H])[.):]?\b", answer, re.IGNORECASE)
    return match[1].upper() if match else "?"


def blink_doc_to_target(doc):
    return _gt_letter(doc)


def _extract_letter(text: str, valid_letters) -> str:
    """Recover the predicted option letter: prefer an answer marker, else a bare letter."""
    cleaned = text or ""
    if "</think>" in cleaned:  # drop a Qwen3.5/Qwen3-VL reasoning block
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


def blink_process_results(doc, results):
    """Score one prediction (exact option-letter match), tagged with its sub-task."""
    pred = results[0] if results else ""
    gt_letter = _gt_letter(doc)
    pred_letter = _extract_letter(pred, _option_letters(doc))
    result = {
        "category": str(doc.get("sub_task") or doc.get("question_type") or "overall"),
        "score": float(pred_letter == gt_letter and pred_letter != ""),
        "pred": pred_letter,
        "gt": gt_letter,
    }
    return {"accuracy": result}


def blink_aggregate_accuracy(results, args=None):
    """Overall multiple-choice accuracy (%); logs the per-sub_task breakdown."""
    if not results:
        return 0.0
    by_category: dict[str, list[float]] = {}
    for result in results:
        by_category.setdefault(result.get("category", "overall"), []).append(
            result["score"]
        )
    for category, scores in sorted(by_category.items()):
        eval_logger.info(
            f"BLINK [{category}]: {sum(scores) / len(scores) * 100:.2f}% "
            f"(n={len(scores)})"
        )
    return float(sum(r["score"] for r in results) / len(results) * 100.0)
