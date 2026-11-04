
"""charades_sta.yaml bindings for the shared TVG implementation."""

from pathlib import Path

from lmms_eval.tasks._task_utils.tvg import (
    tvg_aggregate_iou_03,
    tvg_aggregate_iou_05,
    tvg_aggregate_iou_07,
    tvg_aggregate_m_iou,
    tvg_doc_to_target,
    tvg_process_results,
    TvgTask,
)

__all__ = [
    "tvg_aggregate_iou_03",
    "tvg_aggregate_iou_05",
    "tvg_aggregate_iou_07",
    "tvg_aggregate_m_iou",
    "tvg_doc_to_target",
    "tvg_doc_to_text",
    "tvg_doc_to_visual",
    "tvg_process_results",
]

_TASK = TvgTask.from_yaml(Path(__file__).with_name("charades_sta.yaml"))
tvg_doc_to_visual = _TASK.doc_to_visual
tvg_doc_to_text = _TASK.doc_to_text
