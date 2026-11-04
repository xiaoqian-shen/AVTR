"""Data builders for reconstruction, visual reflection, and supervised fine-tuning."""

from __future__ import annotations


def build_data(cfg: dict, processor, model):
    """-> (IterableDataset, collator) for ``cfg["data"]``."""
    from .media import _qwen_vl_utils

    _qwen_vl_utils()  # fail here: the streams' per-sample handlers would skip every row forever
    data = cfg["data"]
    res = cfg["model"]["resolution"]
    if data["kind"] == "recon":
        from .recon import ReconCollator, ReconStream

        return ReconStream(data, processor, res, seed=data["seed"]), ReconCollator(model.placeholder_id)
    if data["kind"] == "imagine":
        from .imagine import ImagineCollator, ImagineStream

        if cfg["train"].get("per_device_train_batch_size", 1) != 1:
            raise ValueError("the imagine stream needs per_device_train_batch_size == 1")
        stream = ImagineStream(data, processor, model.num_queries, bool(cfg["model"]["markers"]), res,
                               seed=data["seed"])
        return stream, ImagineCollator(processor.tokenizer.pad_token_id)
    raise ValueError(f"unknown data kind {data['kind']!r}")
