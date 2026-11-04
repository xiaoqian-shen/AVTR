
"""qvhighlights.yaml bindings for the shared TVG implementation.

QVHighlights is moment retrieval: the GT `relevant_windows` can hold several disjoint windows,
and the official benchmark scores ranked proposals with mAP. An LLM emits a single interval, so
this reports R@1 IoU (max IoU over the GT windows), the same metric as ActivityNet-TVG.
"""

import fcntl
import os
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.offline_dataset import check_media_cache, stream_extract_members
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
    "qvhighlights_process_docs",
]

# The authors' raw-video archive (all splits): https://github.com/jayleicn/moment_detr#readme
OFFICIAL_VIDEO_ARCHIVE = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"

_TASK = TvgTask.from_yaml(Path(__file__).with_name("qvhighlights.yaml"))
tvg_doc_to_visual = _TASK.doc_to_visual
tvg_doc_to_text = _TASK.doc_to_text
# QVHIGHLIGHTS_VIDEO_DIR: where the val videos live (<vid>.mp4); default $HF_HOME/qvhighlights.
if os.environ.get("QVHIGHLIGHTS_VIDEO_DIR"):
    _TASK.video_dir = os.path.expanduser(os.environ["QVHIGHLIGHTS_VIDEO_DIR"])


def qvhighlights_process_docs(dataset):
    """Make sure every val video is on disk before the run starts.

    Videos missing from the video dir are streamed out of the raw-video archive
    (``QVHIGHLIGHTS_VIDEO_ARCHIVE``: a local path or URL, default the authors' archive): it is
    read once and only the val videos are written, a fraction of it. A file lock lets one
    process do it while the other ranks wait, and the videos already written are kept if a run
    is interrupted (a rerun reads the archive again from the start). The members are
    ``videos/<vid>.mp4`` in the official archive and ``<vid>.mp4`` in a flat one.
    """
    video_dir = _TASK.video_dir
    vids = sorted(set(dataset["vid"]))

    def missing() -> list[str]:
        return [v for v in vids if not os.path.isfile(os.path.join(video_dir, f"{v}.mp4"))]

    if not missing():
        return dataset
    check_media_cache(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    lock = os.open(video_dir.rstrip("/") + ".lock", os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = missing()
        if todo:
            source = os.environ.get("QVHIGHLIGHTS_VIDEO_ARCHIVE", OFFICIAL_VIDEO_ARCHIVE)
            if not source.startswith(("http://", "https://")):
                source = os.path.expanduser(source)
            eval_logger.info(f"qvhighlights: extracting {len(todo)} val videos from {source} into {video_dir}")
            wanted = {}
            for v in vids:  # all of them, so the layout check sees the ones already extracted too
                dest = os.path.join(video_dir, f"{v}.mp4")
                wanted[f"videos/{v}.mp4"] = dest
                wanted[f"{v}.mp4"] = dest
            absent = stream_extract_members(source, wanted)
            if absent:
                raise FileNotFoundError(
                    f"{len(absent)} val videos are not in {source}, e.g. {os.path.basename(absent[0])}"
                )
    finally:
        os.close(lock)
    return dataset
