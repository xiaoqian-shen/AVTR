import os
import re
import sys
from pathlib import Path

import yaml
from lmms_eval.tasks._task_utils.tvg import parse_timestamps_from_string


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "nextgqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# Pass in video path here
# Can only work correctly with video llm
def gqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "nextgqa_videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    options = "\n".join(
        [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(doc["candidates"])]
    )

    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # Match the g_videoqa SFT framing byte-for-byte ("{question}\n{options}\n{instruction}",
    # with NO "Question:"/"Options:" headers) so the model reproduces the trained combined
    # grounding+option answer instead of collapsing to a single task.
    return f"{question}\n{options}\n{post_prompt}"


def gqa_doc_to_answer(doc):
    return {
        "answer": chr(ord("A") + doc["candidates"].index(doc["answer"])),
        "segment": doc["solution"],
    }


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]


def compute_iou(gt_timestamps, pred_timestamps, eps: float = 1e-9):
    # Compute IoU for single intervals
    gs, ge = gt_timestamps[0], gt_timestamps[1]
    ps, pe = pred_timestamps[0], pred_timestamps[1]

    if ge < gs:
        gs, ge = ge, gs

    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), its IoU is zero
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return 0.0

    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter
    return 0.0 if union <= eps else inter / union


def gqa_process_results_generation(doc, result):
    pred = result[0]
    # Parse the option letter from EITHER output format the model may emit:
    #   - compact "answer <> [start, end]" (letter immediately before "<>"), or
    #   - legacy grounding-first "From <t>..</t>s.\nCorrect Option: X".
    # Fall back to the leading A-E heuristic when neither marker is present. The interval is parsed
    # by parse_timestamps_from_string, which already handles both "[a, b]" and the <t> tags.
    option = re.search(r"\b([A-E])\s*<>", pred) or re.search(
        r"correct\s*option\s*[:\-]?\s*\**\s*\(?([A-E])", pred, re.IGNORECASE
    )
    pred_answer = option.group(1).upper() if option else extract_characters_regex(pred)
    pred_timestamps = parse_timestamps_from_string(pred)
    if pred_timestamps is None:
        pred_timestamps = [0.0, 0.0]
    gt_answer = chr(ord("A") + doc["candidates"].index(doc["answer"]))

    gt_timestamps = doc["solution"]
    gt_timestamps = [[float(gt[0]), float(gt[1])] for gt in gt_timestamps]

    result = {
        "query": f"{doc['video']}>>>{doc['candidates']}",
        "gt_answer": gt_answer,
        "gt_segment": gt_timestamps,
        "pred": pred,
        "score": float(gt_answer == pred_answer),
        "iou": max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps]),
    }
    return {
        "accuracy": result,
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result["iou"])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))


def temporal_grounding_aggregate_nextgqa_iou_03(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.3)


def temporal_grounding_aggregate_nextgqa_iou_05(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.5)


def temporal_grounding_aggregate_nextgqa_iou_07(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.7)


def temporal_grounding_aggregate_nextgqa_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result["iou"])

    return float(sum(ious) * 100 / len(ious))


def multi_choice_aggregate_nextgqa_accuracy(results, args):
    accuracy = []
    for result in results:
        accuracy.append(result["score"])

    return float(sum(accuracy) * 100 / len(accuracy))
