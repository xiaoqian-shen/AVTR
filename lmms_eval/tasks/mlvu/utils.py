import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mlvu.yaml", "r") as f:
    raw_data_mavg = f.readlines()
    safe_data_mavg = []
    for line in raw_data_mavg:
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_mavg.append(line)
cache_name_mavg = yaml.safe_load("".join(safe_data_mavg))["dataset_kwargs"]["cache_dir"]
cache_dir_mavg = os.path.join(base_cache_dir, cache_name_mavg)

# The multiple-choice task types of MLVU M-Avg (mlvu.yaml), in the order MLVU reports them.
MAVG_TASK_TYPES = [
    "topic_reasoning",
    "anomaly_reco",
    "needle",
    "ego",
    "count",
    "order",
    "plotQA",
]


def mlvu_mavg_doc_to_visual(doc):
    """The video zips extract flat, so the video is <cache dir>/<video_name>."""
    video_path = os.path.join(cache_dir_mavg, doc["video_name"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path:{video_path} does not exist.")
    return [video_path]


def mlvu_mavg_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """`question` already carries the lettered options inline ("...?\\n(A) ...\\n(B) ..."), so
    the unlettered `candidates` are not appended: rendering every option twice silently tanks a
    multiple-choice score."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question'].rstrip()}\n{post_prompt}"


def mlvu_mavg_parse_choice(response, all_choices):
    """Letter out of a free-form answer, mirroring longvideobench's parser: an unparseable answer
    gets a random choice instead of counting as wrong. The draw is unseeded, so such rows can
    flip between runs."""
    s = (response or "").strip()
    for prefix in (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ):
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search(r"[ABCD]", s):
        return random.choice(all_choices)
    match = re.search(r"[ABCD]", s)
    if match is None:
        return random.choice(all_choices)
    return match[0]


def mlvu_mavg_process_results(doc, results):
    n_choices = len(doc.get("candidates") or []) or 4
    all_choices = [chr(ord("A") + i) for i in range(n_choices)]
    parsed_pred = mlvu_mavg_parse_choice(results[0], all_choices)
    # `answer` is already a letter in this dataset, not an index into candidates.
    gold = str(doc["answer"]).strip().upper()[:1]
    return {
        "mlvu_m_avg": {
            "question_id": doc.get("question_id"),
            "task_type": doc.get("task_type"),
            "answer": gold,
            "parsed_pred": parsed_pred,
            "score": float(parsed_pred == gold),
        }
    }


def mlvu_mavg_aggregate_results(results):
    """Per-type accuracies plus M-Avg, as percentages. M-Avg is the instance-level mean, not
    the mean of the per-type means: the types are unbalanced, and the instance-level number is
    what MLVU's leaderboard reports."""
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r["score"])

    known_types = [t for t in MAVG_TASK_TYPES if by_type.get(t)]
    extra_types = sorted(set(by_type) - set(MAVG_TASK_TYPES))
    printable = {}
    for t in known_types + extra_types:
        scores = by_type[t]
        printable[t] = {
            "num": len(scores),
            "acc": round(100 * sum(scores) / len(scores), 2),
        }

    total = [s for scores in by_type.values() for s in scores]
    overall = 100 * sum(total) / len(total) if total else 0.0
    printable["Overall"] = {"num": len(total), "acc": round(overall, 2)}
    eval_logger.info(printable)
    return overall
