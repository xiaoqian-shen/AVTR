"""Normalize JoyAI, Open-o3, VideoKR, and SpatialEdit shard records."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from typing import Any

LLAVA_IMAGE = "<image>"
LLAVA_VIDEO = "<video>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"


def clue_key(media_bytes: bytes, question: str) -> str:
    """Content-addressed sidecar key: blake2b(media bytes) : blake2b(question.strip()), short hex digests."""
    media = hashlib.blake2b(media_bytes, digest_size=8).hexdigest()
    q = hashlib.blake2b(question.strip().encode("utf-8"), digest_size=6).hexdigest()
    return f"{media}:{q}"


# ---------------------------------------------------------------------------------------------
# Raw record schemas
# ---------------------------------------------------------------------------------------------
_O3_FORMAT = (
    "A conversation between user and assistant. The user provides a {media} and asks "
    "a question, and the Assistant solves it. The assistant MUST first think about "
    "the reasoning process in the mind and then provide the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and <answer> "
    "</answer> tags, respectively."
)
# Open-o3-Video per-task (system prompt, user-question prefix), verbatim from upstream.
_O3_TASKS: dict[str, tuple[str, str]] = {
    "visual QA": (
        _O3_FORMAT.format(media="image")
        + " When referring to particular objects in the reasoning process, the "
        "assistant MUST localize the object with bounding box coordinates between "
        "<box> and </box>. You MUST strictly follow the format.",
        "",
    ),
    "temporal-spatial free-form QA": (
        _O3_FORMAT.format(media="video")
        + " All reasoning must be grounded in visual evidence from the video. When "
        "you mention any related object, person, or specific visual element, you must "
        "strictly follow the following format: `<obj>object_name</obj><box>"
        "bounding_box</box>at<t>time_in_seconds</t>s`.",
        "",
    ),
    "temporal QA": (
        "A conversation between user and assistant. The user provides a video and "
        "asks a question, and the Assistant determines the precise time period that "
        "answers the question. The assistant MUST first think about the reasoning "
        "process in the mind and then provide the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> "
        "</answer> tags, respectively. The answer must strictly follow the following "
        "format: `From <t>start_time</t>s to <t>end_time</t>s'",
        "Question: ",
    ),
    "temporal QA (MCQ)": (
        "A conversation between user and assistant. The user provides a video and a "
        "multiple-choice question, and the Assistant determines the precise time "
        "period that answers the question and selects the correct option. The "
        "assistant MUST first think about the reasoning process in the mind and then "
        "provide the user with the answer. The reasoning process and answer are "
        "enclosed within <think> </think> and <answer> </answer> tags, respectively. "
        "When mentioning time during the reasoning process, the assistant must use "
        "the format: `<t>time_in_seconds</t>s'. The answer must strictly follow the "
        "following format: `From <t>start_time</t>s to <t>end_time</t>s.\n"
        "Correct Option: [ONLY THE LETTER]'.",
        "Question: ",
    ),
    "General video QA MCQ": (
        _O3_FORMAT.format(media="video").replace("asks a question", "asks a multiple-choice question")
        + " Only output the correct option in the <answer> </answer> section.",
        "Question: ",
    ),
    "General video QA Free-form": (_O3_FORMAT.format(media="video"), "Question: "),
}

_BOX_RE = re.compile(r"<box>\s*\[([^\]]+)\]\s*</box>")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def _rescale_boxes(text: str, image_size=None) -> str:
    """``<box>[...]</box>`` -> Qwen's resolution-independent integer grid: video boxes are normalized
    coordinates; image boxes are pixels (divided by the image size first). Malformed boxes are left untouched."""
    if image_size is not None:
        w, h = float(image_size[0]), float(image_size[1])
        scales = (1000.0 / w, 1000.0 / h, 1000.0 / w, 1000.0 / h)
    else:
        scales = (1000.0,) * 4

    def repl(m: re.Match) -> str:
        parts = m.group(1).split(",")
        if len(parts) != 4:
            return m.group(0)
        try:
            coords = [float(p) for p in parts]
        except ValueError:
            return m.group(0)
        grid = (str(min(1000, max(0, round(c * s)))) for c, s in zip(coords, scales, strict=True))
        return f"<box>[{','.join(grid)}]</box>"

    return _BOX_RE.sub(repl, text)


def _o3_record(raw: dict) -> dict:
    task = str(raw.get("task") or "")
    if task not in _O3_TASKS:
        raise ValueError(f"unknown Open-o3-Video task {task!r}")
    system, prefix = _O3_TASKS[task]
    video_path = str(raw.get("video_path") or "").strip()
    media = LLAVA_VIDEO if video_path else LLAVA_IMAGE
    reasoning = str(raw.get("reasoning_process") or "").strip()
    answer = str(raw.get("answer") or "").strip()
    if video_path:
        reasoning, answer = _rescale_boxes(reasoning), _rescale_boxes(answer)
    else:
        size = raw.get("image_size")
        if size and len(size) == 2 and size[0] and size[1]:
            reasoning, answer = _rescale_boxes(reasoning, size), _rescale_boxes(answer, size)
    answer = f"<answer>{answer}</answer>"
    value = f"<think>{reasoning}</think>\n{answer}" if reasoning else answer
    rec: dict[str, Any] = {
        "conversations": [{"from": "human", "value": f"{media}\n{prefix}{str(raw['question']).strip()}"},
                          {"from": "gpt", "value": value}],
        "system": system,
    }
    if video_path:
        rec["video"] = video_path
    elif raw.get("image_path"):
        rec["image"] = raw["image_path"]
    return rec


def normalize_record(raw: dict) -> dict:
    if "conversations" in raw:
        return raw
    if "reasoning_process" in raw:
        return _o3_record(raw)
    if "question" in raw and "answer" in raw:
        answer = str(raw["answer"]).strip()
        # An inline leading <think>...</think> is dropped from the target (reasoning is not trained).
        m = re.match(r"\s*<think>(.*?)</think>\s*(.*)\Z", answer, flags=re.DOTALL)
        if m:
            answer = m.group(2).strip()
        rec = {"conversations": [{"from": "human", "value": f"{LLAVA_VIDEO}\n{str(raw['question']).strip()}"},
                                 {"from": "gpt", "value": answer}]}
        for key in ("video", "image"):
            if raw.get(key):
                rec[key] = raw[key]
        return rec
    if "file_name" in raw and "question" in raw and "solution" in raw:
        human = f"{LLAVA_VIDEO}\n{str(raw['question']).strip()}"
        if raw.get("options"):
            human += "\n" + "\n".join(str(o).strip() for o in raw["options"])
        path = str(raw["file_name"]).strip()
        path = path[2:] if path.startswith("./") else path
        key = "image" if any(path.lower().endswith(e) for e in _IMAGE_EXTS) else "video"
        return {"conversations": [{"from": "human", "value": human},
                                  {"from": "gpt", "value": str(raw["solution"]).strip()}], key: path}
    raise ValueError(f"unsupported record schema: keys {sorted(raw)}")


def normalize_image_placeholders(conversations: list[dict], n_images: int) -> None:
    """One ``<image>`` per image: strip vision wrappers, fold raw ``<|image_pad|>`` to ``<image>``,
    then prepend the still-missing ones to the first human turn (JoyAI mixes these styles)."""
    for turn in conversations:
        v = (turn.get("value") or "").replace(VISION_START, "").replace(VISION_END, "")
        turn["value"] = v.replace(IMAGE_PAD, LLAVA_IMAGE)
    missing = n_images - sum(t.get("value", "").count(LLAVA_IMAGE) for t in conversations)
    if missing > 0:
        for turn in conversations:
            if turn.get("from") == "human":
                turn["value"] = f"{LLAVA_IMAGE}\n" * missing + turn["value"]
                return


def normalize_video_placeholder(conversations: list[dict]) -> None:
    """Exactly one ``<video>`` on a video row: drop image placeholders (a packer bug on video rows),
    fold raw ``<|video_pad|>``, prepend one if none is left."""
    for turn in conversations:
        v = (turn.get("value") or "").replace(VISION_START, "").replace(VISION_END, "").replace(VIDEO_PAD, LLAVA_VIDEO)
        for tok in (IMAGE_PAD, LLAVA_IMAGE):
            v = v.replace(tok, "")
        turn["value"] = v.lstrip("\n")
    if any(t.get("value", "").count(LLAVA_VIDEO) for t in conversations):
        return
    for turn in conversations:
        if turn.get("from") == "human":
            turn["value"] = f"{LLAVA_VIDEO}\n{turn['value']}"
            return


def decode_pil(raw: Any):
    from PIL import Image

    data = raw.get("bytes") if isinstance(raw, dict) else raw
    if data is None and isinstance(raw, dict) and raw.get("path"):
        with open(str(raw["path"]), "rb") as fh:
            data = fh.read()
    if not data:
        raise ValueError("empty image element")
    return Image.open(io.BytesIO(bytes(data))).convert("RGB")


# ---------------------------------------------------------------------------------------------
# Row adapters
# ---------------------------------------------------------------------------------------------
def adapt_joyai(row: dict) -> list[dict]:
    convs = row.get("conversations")
    if not convs:
        return []
    rec: dict[str, Any] = {"conversations": [{"from": t["from"], "value": t["value"]} for t in convs],
                           "id": row.get("id")}
    images = row.get("images") or []
    if images:
        rec["image"] = [decode_pil(im) for im in images]
        normalize_image_placeholders(rec["conversations"], len(images))
    return [rec]


def adapt_open_o3(row: dict) -> list[dict]:
    raw = json.loads(row["record_json"])
    rec = normalize_record(raw)
    bbox = raw.get("bbox")  # raw-pixel xyxy boxes (VisualCoT)
    if bbox:
        rec["bbox"] = bbox
    if "video" in rec:
        if not row.get("video"):
            return []
        rec["_video_bytes"] = bytes(row["video"])
        normalize_video_placeholder(rec["conversations"])
    elif "image" in rec:
        images = row.get("images") or []
        if not images:
            return []
        rec["image"] = [decode_pil(im) for im in images]
        if bbox:
            # Region-sidecar key: the ORIGINAL encoded bytes + the raw question, as the washer hashed.
            conv = raw.get("conversations") or []
            if conv:
                q = str(conv[0].get("value", "")).replace(LLAVA_IMAGE, "").strip()
                rec["_region_key"] = clue_key(bytes(images[0]), q)
    return [rec]


def adapt_videokr(row: dict) -> list[dict]:
    """One video + an array of its QA records -> one record per QA, all sharing the video bytes."""
    if not row.get("video"):
        return []
    payload = bytes(row["video"])
    raws = json.loads(row["record_json"])
    out = []
    for raw in [raws] if isinstance(raws, dict) else raws:
        rec = normalize_record(raw)
        if "video" not in rec:
            continue
        rec["_video_bytes"] = payload
        if raw.get("spans"):  # pre-parsed [start, end] seconds (answers without <t> markup)
            rec["spans"] = raw["spans"]
        out.append(rec)
    return out


def adapt_spatialedit(row: dict) -> list[dict]:
    """The source goes to ``image`` (the backbone sees it); the edited target rides in a private key."""
    try:
        source = decode_pil(row["source_bytes"])
    except Exception:  # noqa: BLE001 -- an undecodable jpeg drops its sample
        return []
    return [{
        "conversations": [{"from": "human", "value": f"{LLAVA_IMAGE}\n{row['instruction']}"},
                          {"from": "gpt", "value": row["instruction"]}],
        "image": [source], "_edit_target_bytes": row["target_bytes"], "id": row["key"],
    }]


ADAPTERS = {"joyai": adapt_joyai, "open_o3": adapt_open_o3, "videokr": adapt_videokr, "spatialedit": adapt_spatialedit}


def read_spatialedit_tar(data: bytes) -> list[dict]:
    """Regroup ``{key}.0.jpg`` / ``{key}.1.jpg`` / ``{key}.json`` members; incomplete groups drop."""
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            for suffix, field in ((".0.jpg", "source_bytes"), (".1.jpg", "target_bytes"), (".json", "json")):
                if m.name.endswith(suffix):
                    fh = tar.extractfile(m)
                    if fh is not None:
                        groups.setdefault(m.name[: -len(suffix)], {})[field] = fh.read()
                    break
    rows = []
    for key, g in groups.items():
        if not {"source_bytes", "target_bytes", "json"} <= g.keys():
            continue
        try:
            turns = json.loads(g["json"]).get("conversations") or []
            instruction = str(turns[0].get("value", "")) if turns else ""
        except Exception:  # noqa: BLE001
            continue
        instruction = instruction.replace(LLAVA_IMAGE, "").strip()
        if instruction:
            rows.append({"key": key, "source_bytes": g["source_bytes"], "target_bytes": g["target_bytes"],
                         "instruction": instruction})
    return rows
