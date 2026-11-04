
"""ERQA (Embodied Reasoning QA) task helpers for lmms_eval.

ERQA (Google DeepMind, https://github.com/embodiedreasoning/ERQA) is a
multiple-choice benchmark probing embodied/spatial reasoning, each question grounded in one
OR MORE images. Questions span the ``question_type`` categories (Spatial Reasoning, Action
Reasoning, Trajectory Reasoning, State Estimation, Task Reasoning, Multi-view Reasoning,
Pointing, Other). Scored on a single axis -- multiple-choice answer accuracy (A-D) --
reported overall and per ``question_type``.

Each ``doc`` (a row of the ``test`` parquet ``data/test-00000-of-00001.parquet`` of the HF repo
``FlagEval/ERQA``) has: ``question_id``, ``question`` (the
preformatted prompt with inline ``Choices:`` and a baked-in "answer with only the letter"
instruction), ``answer`` (the option letter), ``question_type`` (category), ``images`` (a
list of HF Image structs), and ``visual_indices`` (the character offset in ``question`` at
which each parallel image is placed). We present the images in that
display order (ascending offset) and prepend them to the text -- the default Qwen3-VL
wrapper path (``interleave_visuals=False``) puts all images before the prompt, and the
questions refer to images positionally ("first image" / "the previous image"), so the order
is what matters. The yaml wires the functions below via ``!function utils.<name>``.
"""

from __future__ import annotations

import base64
import io
import re

from loguru import logger as eval_logger
from PIL import Image

# An ERQA option label: a bare A-D letter followed by ".", ")" and a space, e.g. "A. " /
# "B) " (choices are inline, not at line starts). Restricted to A-D since no ERQA question has more options.
_OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-D])[.)]\s")


def _to_pil(raw: object) -> Image.Image:
    """Decode one image element (PIL / HF ``{bytes,path}`` / bytes / base64 / path) to RGB."""
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


def _ordered_images(doc: dict) -> list:
    """Return the doc's images in display order (ascending ``visual_indices`` offset).

    ``images[i]`` is placed at character offset ``visual_indices[i]`` in the question, so the
    display order is a stable sort by offset. Falls back to raw list order when there is no
    usable parallel ``visual_indices`` (e.g. single-image rows store an empty list).
    """
    images = doc.get("images")
    if images is None:
        images = doc.get("image")
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        return [images]
    images = list(images)
    indices = doc.get("visual_indices")
    try:
        indices = list(indices) if indices is not None else None
    except TypeError:
        indices = None
    if indices is not None and len(indices) == len(images):
        order = sorted(range(len(images)), key=lambda i: indices[i])
        return [images[i] for i in order]
    return images


def erqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Return the question's images as a list of PIL.Image (the model base64-encodes them)."""
    return [_to_pil(img) for img in _ordered_images(doc)]


def erqa_process_docs(dataset):
    """Cast the ``images`` column to an HF ``Sequence(Image)`` feature (decode + log strip).

    Run at load time (yaml ``process_docs``) so (a) ``doc_to_visual`` gets decoded PIL
    images and (b) lmms_eval drops them from the ``--log_samples`` copy -- its
    ``dataset_no_image`` only strips ``Image``/``Audio`` features, so a raw struct column
    would otherwise be stringified into huge sample logs. No-op if already an Image
    sequence or absent.
    """
    import datasets

    feature = getattr(dataset, "features", {}).get("images")
    already_image = isinstance(feature, datasets.Sequence) and isinstance(
        feature.feature, datasets.Image
    )
    if feature is not None and not already_image:
        dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    return dataset


def erqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Render the prompt text: optional pre-prompt, the question (choices inline), post-prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    return "\n".join(part for part in (pre_prompt, question, post_prompt) if part)


def _gt_letter(doc: dict) -> str:
    """Map the ground-truth answer to its option letter (A-D), or ``"?"``."""
    answer = str(doc.get("answer", "")).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    match = re.match(r"\(?([A-D])[.):]?\b", answer, re.IGNORECASE)
    return match[1].upper() if match else "?"


def erqa_doc_to_target(doc):
    return _gt_letter(doc)


def _option_letters(doc: dict) -> list[str]:
    """Valid option letters: scanned from inline ``Choices:`` labels, else A-D."""
    found = list(dict.fromkeys(_OPTION_LABEL_RE.findall(str(doc.get("question", "")))))
    return found or list("ABCD")


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


def erqa_process_results(doc, results):
    """Score one prediction (exact option-letter match), tagged with its question_type."""
    pred = results[0] if results else ""
    gt_letter = _gt_letter(doc)
    pred_letter = _extract_letter(pred, _option_letters(doc))
    result = {
        "category": str(doc.get("question_type") or "overall"),
        "score": float(pred_letter == gt_letter and pred_letter != ""),
        "pred": pred_letter,
        "gt": gt_letter,
    }
    return {"accuracy": result}


def erqa_aggregate_accuracy(results, args=None):
    """Overall multiple-choice accuracy (%); logs the per-question_type breakdown."""
    if not results:
        return 0.0
    by_category: dict[str, list[float]] = {}
    for result in results:
        by_category.setdefault(result.get("category", "overall"), []).append(
            result["score"]
        )
    for category, scores in sorted(by_category.items()):
        eval_logger.info(
            f"ERQA [{category}]: {sum(scores) / len(scores) * 100:.2f}% "
            f"(n={len(scores)})"
        )
    return float(sum(r["score"] for r in results) / len(results) * 100.0)
