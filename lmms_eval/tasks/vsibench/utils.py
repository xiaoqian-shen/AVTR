import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_final_boxed_content,
)
from loguru import logger as eval_logger

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "")
        return question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "")
        return "\n".join([question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def _strip_reasoning(pred):
    text = pred or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _extract_mca(pred):
    """Option letter: an explicit answer marker first (the LAST one, since a reasoning turn names
    several options before concluding), else the first bare letter, else `fuzzy_matching`.

    Both patterns accept lowercase a-d, so an English article counts as option A: "It is a
    chair" -> "A", and "the answer is a chair" -> "A".
    """
    text = _strip_reasoning(pred)
    marked = re.findall(
        r"(?:answer|option|choice)\b\s*(?:is|are|:|=)?\s*\(?([A-Da-d])\)?\b", text
    )
    if marked:
        return marked[-1].upper()
    bare = re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
    if bare:
        return bare[0].upper()
    return fuzzy_matching(pred)


def _extract_na(pred):
    """First number anywhere, not just the first word, with commas stripped so thousands separators
    parse. `fuzzy_matching` keeps only the first word, so a number after a leading word scores nothing."""
    text = _strip_reasoning(pred).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def _score_doc(doc, extract_mca, extract_na):
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(extract_mca(doc["prediction"]), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(
                    to_float(extract_na(doc["prediction"])),
                    to_float(doc["ground_truth"]),
                )
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    return {"vsibench_score": doc}


def vsibench_process_results(doc, results):
    doc["prediction"] = results[0]
    return _score_doc(doc, _extract_mca, _extract_na)


def vsibench_process_results_boxed(doc, results):
    doc["prediction"] = extract_final_boxed_content(results[0])
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(
                fuzzy_matching(doc["prediction"]), doc["ground_truth"]
            )
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(
                    to_float(fuzzy_matching(doc["prediction"])),
                    to_float(doc["ground_truth"]),
                )
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}


def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby(
        "question_type"
    ).groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == "success_rate":
                    output[f"{question_type}_{metric}"] = per_question_type[
                        metric
                    ].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[
                        metric
                    ].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output["object_rel_direction_accuracy"] = (
        sum(
            [
                output.pop("object_rel_direction_easy_accuracy"),
                output.pop("object_rel_direction_medium_accuracy"),
                output.pop("object_rel_direction_hard_accuracy"),
            ]
        )
        / 3.0
    )

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output
