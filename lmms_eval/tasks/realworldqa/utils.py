
"""RealWorldQA task helpers for lmms_eval.

RealWorldQA (X.AI, https://huggingface.co/datasets/xai-org/RealworldQA / the
``lmms-lab`` mirror) is a set of single-image real-world spatial-understanding questions.
It is MIXED-format: some are multiple choice (answer is an option letter), some are
open-ended (answer is a word or number, e.g. a count or ``"Downhill"``). The question
text already carries the answer-format instruction ("Please answer directly with ...").

Each ``doc`` (a row of the ``test`` parquets, ``data/test-*.parquet`` of the HF mirror named in
the yaml) has: ``question`` (instruction inline),
``answer`` (the gold letter / word / number), ``image`` (one HF Image struct). The yaml
wires the functions below via ``!function utils.<name>``. Scored on answer accuracy:
exact option-letter match for the MC questions, normalized string/number match otherwise.
"""

from __future__ import annotations

import base64
import io
import re

from loguru import logger as eval_logger
from PIL import Image

# An option label at the start of a line: "A. ", "B) ", "(C) " (optionally parenthesized).
_OPTION_LABEL_RE = re.compile(r"(?m)^\s*\(?([A-H])[.):]\s+")


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


def realworldqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Return the question's single image as a one-element list of PIL.Image."""
    image = doc.get("image")
    if image is None:
        image = doc.get("images")
    if isinstance(image, (list, tuple)):
        return [_to_pil(img) for img in image]
    return [] if image is None else [_to_pil(image)]


def realworldqa_process_docs(dataset):
    """Cast the ``image`` column to an HF ``Image`` feature (decode + strip from log_samples).

    Mirrors the mmsi/blink trick: lmms_eval's ``dataset_no_image`` only drops ``Image``/
    ``Audio`` features from the ``--log_samples`` copy, so leaving ``image`` as a raw
    struct would stringify every sample's image bytes into the logs. No-op if already Image.
    """
    import datasets

    feature = getattr(dataset, "features", {}).get("image")
    if feature is not None and not isinstance(feature, datasets.Image):
        dataset = dataset.cast_column("image", datasets.Image())
    return dataset


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Render the prompt: optional pre-prompt, the question (instruction inline), post-prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    return "\n".join(part for part in (pre_prompt, question, post_prompt) if part)


def realworldqa_doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def _strip_think(text: str) -> str:
    cleaned = text or ""
    if "</think>" in cleaned:  # drop a Qwen3.5/Qwen3-VL reasoning block
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.strip()


def _is_mc(doc: dict, gt: str) -> bool:
    """A multiple-choice item: gold is a single letter AND the question lists options."""
    return (
        len(gt) == 1
        and gt.isalpha()
        and bool(_OPTION_LABEL_RE.search(str(doc.get("question", ""))))
    )


def _option_letters(doc: dict) -> list[str]:
    found = list(dict.fromkeys(_OPTION_LABEL_RE.findall(str(doc.get("question", "")))))
    return found or list("ABCD")


def _extract_letter(text: str, valid_letters) -> str:
    """Recover the predicted option letter: prefer an answer marker, else a bare letter."""
    cleaned = _strip_think(text)
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


def _normalize(text: str) -> str:
    """Lowercase, drop a leading 'the answer is', strip surrounding punctuation/space."""
    cleaned = _strip_think(text).lower()
    cleaned = re.sub(r"^\s*(the\s+)?(answer|it)\s+(is|are)\s*[:=]?\s*", "", cleaned)
    cleaned = cleaned.strip().strip(".!?,:;\"'()[]")
    return re.sub(r"\s+", " ", cleaned).strip()


def _open_match(pred: str, gt: str) -> bool:
    """Normalized match for an open-ended (word/number) answer."""
    np_, ng = _normalize(pred), _normalize(gt)
    if not ng:
        return False
    if np_ == ng:
        return True
    # gt as a standalone token in the prediction ("...is downhill" vs "Downhill"); the
    # word-boundary split avoids a short number matching inside a longer one.
    if ng in re.split(r"[\s/]+", np_):
        return True
    # numeric equality (an integer vs the same value written as a decimal)
    try:
        return float(np_) == float(ng)
    except ValueError:
        return False


def realworldqa_process_results(doc, results):
    """Score one prediction; tag as multiple_choice vs open for the breakdown."""
    pred = results[0] if results else ""
    gt = str(doc.get("answer", "")).strip()
    if _is_mc(doc, gt):
        category = "multiple_choice"
        score = float(_extract_letter(pred, _option_letters(doc)) == gt.upper())
    else:
        category = "open"
        score = float(_open_match(pred, gt))
    return {"accuracy": {"category": category, "score": score, "pred": pred, "gt": gt}}


def realworldqa_aggregate_accuracy(results, args=None):
    """Overall accuracy (%); logs the multiple_choice vs open breakdown."""
    if not results:
        return 0.0
    by_category: dict[str, list[float]] = {}
    for result in results:
        by_category.setdefault(result.get("category", "overall"), []).append(
            result["score"]
        )
    for category, scores in sorted(by_category.items()):
        eval_logger.info(
            f"RealWorldQA [{category}]: {sum(scores) / len(scores) * 100:.2f}% "
            f"(n={len(scores)})"
        )
    return float(sum(r["score"] for r in results) / len(results) * 100.0)
