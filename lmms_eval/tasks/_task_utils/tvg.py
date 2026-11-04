
"""Temporal video grounding (TVG): localize a text query in a video as [start, end] seconds.

The interval is parsed from the model's free text and scored against the best-matching
ground-truth window as R@1 at several IoU thresholds plus mIoU, the same metrics as nextgqa's
grounding side, so the curves are comparable. Each task directory binds `TvgTask` to its OWN
yaml by file name: one directory can hold several task yamls with different cache dirs.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

# Field-name variants across TVG dumps (ActivityNet-TVG, Charades-STA, QVHighlights).
VIDEO_KEYS = ("video", "vid", "video_id", "video_name", "video_path")
QUERY_KEYS = ("query", "sentence", "caption", "text", "question")
TIMESTAMP_KEYS = (
    "timestamps",
    "relevant_windows",
    "gt_windows",
    "timestamp",
    "gt",
    "span",
)
DURATION_KEYS = ("duration", "video_duration", "vid_duration")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")

_NUM = r"(?:\d+(?:\.\d+)?)"
_SEP = r"(?:,|-|–|—|\bto\b|\band\b)"
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<t>\s*(\d+(?:\.\d+)?)\s*</t>")
_INTERVAL_RE = re.compile(
    rf"""
    (?<![\d.])
    (?:
        \[\s*(?P<sb>{_NUM})\s*{_SEP}\s*(?P<eb>{_NUM})\s*\]
      | \(\s*(?P<sp>{_NUM})\s*{_SEP}\s*(?P<ep>{_NUM})\s*\)
      | (?P<s>{_NUM})\s*{_SEP}\s*(?P<e>{_NUM})
    )
    (?![\d.])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LOOSE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def load_dataset_kwargs(yaml_path: str | os.PathLike[str]) -> dict[str, Any]:
    """`dataset_kwargs` of one task yaml; `!function` lines are dropped since safe_load rejects
    the tag."""
    with open(yaml_path, "r") as f:
        safe_lines = [line for line in f if "!function" not in line]
    return yaml.safe_load("".join(safe_lines))["dataset_kwargs"]


def _first(doc: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in doc and doc[key] is not None:
            return doc[key]
    return default


def _video_id(doc: Mapping[str, Any]) -> str:
    return str(_first(doc, VIDEO_KEYS, ""))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _window(pair: Sequence[Any]) -> list[float] | None:
    start, end = _to_float(pair[0]), _to_float(pair[1])
    return [start, end] if start is not None and end is not None else None


def gt_windows(doc: Mapping[str, Any]) -> list[list[float]]:
    """Ground-truth window(s) as a list of [start, end] floats, one or many.

    Endpoints go through float() instead of an isinstance gate: HF returns a Sequence(float16)
    column as np.float16 and some dumps store decimal strings. An isinstance check silently
    dropped such windows, leaving an empty GT that scored every row as a miss.
    """
    raw = _first(doc, TIMESTAMP_KEYS, [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    if not isinstance(raw[0], (list, tuple)):
        window = _window(raw) if len(raw) >= 2 else None
        return [window] if window else []
    windows = [
        _window(win) for win in raw if isinstance(win, (list, tuple)) and len(win) >= 2
    ]
    return [win for win in windows if win]


def _ordered(start: str, end: str) -> list[float]:
    low, high = float(start), float(end)
    return [low, high] if low <= high else [high, low]


def parse_timestamps_from_string(pred_text: str | None) -> list[float] | None:
    """First [start, end] interval in seconds (start <= end) in a model answer, or None.

    Preference order: the <t>..</t> training format ("From <t>a</t>s to <t>b</t>s",
    optionally inside <answer>..</answer>), taking the first and last tag; then [a, b], (a, b)
    or a, b with , - – — "to" "and" separators (plain numbers, no mm:ss clocks); then the first
    two bare numbers. The last step recovers broken tags ("<t>=b</t>", "<t><a<t>") and
    unit-glued spans ("as to bs"), which otherwise scored many correct spans as misses.
    """
    text = (pred_text or "").strip()
    answer = _ANSWER_RE.search(text)
    if answer:
        text = answer.group(1).strip()
    tags = _TAG_RE.findall(text)
    if len(tags) >= 2:
        return _ordered(tags[0], tags[-1])
    match = _INTERVAL_RE.search(text)
    if match:
        return _ordered(
            match.group("sb") or match.group("sp") or match.group("s"),
            match.group("eb") or match.group("ep") or match.group("e"),
        )
    loose = _LOOSE_NUMBER_RE.findall(text)
    if len(loose) >= 2:
        return _ordered(loose[0], loose[1])
    return None


def compute_iou(gt: Sequence[Any], pred: Sequence[Any], eps: float = 1e-9) -> float:
    gs, ge = sorted((float(gt[0]), float(gt[1])))
    ps, pe = sorted((float(pred[0]), float(pred[1])))
    len_p = max(0.0, pe - ps)
    if len_p <= eps:  # a point prediction overlaps nothing
        return 0.0
    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter
    return 0.0 if union <= eps else inter / union


def tvg_doc_to_target(doc: Mapping[str, Any]) -> list[list[float]]:
    return gt_windows(doc)


def tvg_process_results(
    doc: Mapping[str, Any], result: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """R@1: IoU of the single predicted interval against the best-matching GT window."""
    pred_text = result[0] if result else ""
    pred = parse_timestamps_from_string(pred_text)
    gts = gt_windows(doc)
    if pred is None or not gts:
        iou = 0.0
    else:
        iou = max(compute_iou(gt, pred) for gt in gts)
    row = {
        "video": _video_id(doc),
        "query": str(_first(doc, QUERY_KEYS, "")),
        "gt": gts,
        "pred_text": pred_text,
        "pred": pred,
        "iou": iou,
    }
    return {"iou_0.3": row, "iou_0.5": row, "iou_0.7": row, "m_iou": row}


def _recall_at_iou(results: Sequence[Mapping[str, Any]], threshold: float) -> float:
    if not results:
        return 0.0
    hit = sum(1 for r in results if r["iou"] >= threshold)
    return float(hit * 100.0 / len(results))


def tvg_aggregate_iou_03(results: Sequence[Mapping[str, Any]], args: Any) -> float:
    return _recall_at_iou(results, 0.3)


def tvg_aggregate_iou_05(results: Sequence[Mapping[str, Any]], args: Any) -> float:
    return _recall_at_iou(results, 0.5)


def tvg_aggregate_iou_07(results: Sequence[Mapping[str, Any]], args: Any) -> float:
    return _recall_at_iou(results, 0.7)


def tvg_aggregate_m_iou(results: Sequence[Mapping[str, Any]], args: Any) -> float:
    if not results:
        return 0.0
    return float(sum(r["iou"] for r in results) * 100.0 / len(results))


def _header_duration(path: str) -> float | None:
    """Container-header duration in seconds, or None when OpenCV cannot tell. Only the header
    is read: decord indexes every frame first, much slower, and this runs once per row."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
    except cv2.error:
        return None
    return frames / fps if fps > 0 and frames > 0 else None


class TvgTask:
    """Video lookup and prompt building for one TVG task yaml."""

    def __init__(
        self, cache_name: str, video_subdir: str, hf_home: str | None = None
    ) -> None:
        root = hf_home or os.getenv("HF_HOME", "~/.cache/huggingface/")
        self.video_dir: str = os.path.join(
            os.path.expanduser(root), cache_name, video_subdir
        )
        self._durations: dict[str, float | None] = {}

    @classmethod
    def from_yaml(
        cls, yaml_path: str | os.PathLike[str], hf_home: str | None = None
    ) -> TvgTask:
        kwargs = load_dataset_kwargs(yaml_path)
        cache_name = kwargs["cache_dir"]
        video_subdir = kwargs.get("video_subdir", f"{cache_name}_videos")
        return cls(cache_name, video_subdir, hf_home)

    def doc_to_visual(
        self,
        doc: Mapping[str, Any],
        lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """The doc's video path; ids may be stored with or without a file extension.

        A missing video raises: the model adapters drop a visual path without a known video
        extension, so returning the bare id would run the row text-only and silently score it as wrong.
        """
        video_path = os.path.join(self.video_dir, _video_id(doc))
        if os.path.exists(video_path):
            return [video_path]
        for ext in VIDEO_EXTENSIONS:
            if os.path.exists(video_path + ext):
                return [video_path + ext]
        raise FileNotFoundError(
            f"no video for {_video_id(doc)!r} in {self.video_dir} (tried the id as is and with "
            f"{'/'.join(VIDEO_EXTENSIONS)})"
        )

    def video_duration(self, doc: Mapping[str, Any]) -> float | None:
        """Seconds: the dataset's own duration field when it has one (QVHighlights), else the
        container header, cached per video."""
        value = _to_float(_first(doc, DURATION_KEYS))
        if value:
            return value
        path = self.doc_to_visual(doc)[0]
        if path not in self._durations:
            self._durations[path] = _header_duration(path)
        return self._durations[path]

    def doc_to_text(
        self,
        doc: Mapping[str, Any],
        lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """post_prompt is one verbatim SFT instruction with a {query} placeholder, so the eval
        prompt matches training. {duration} is opt-in: prompts without it never open the video."""
        kwargs = lmms_eval_specific_kwargs or {}
        query = str(_first(doc, QUERY_KEYS, "")).strip()
        text = f"{kwargs.get('pre_prompt', '')}{kwargs.get('post_prompt', '')}"
        text = text.replace("{query}", query)
        if "{duration}" in text:
            seconds = self.video_duration(doc)
            text = text.replace(
                "{duration}", f"{seconds:.1f}" if seconds else "unknown"
            )
        return text
