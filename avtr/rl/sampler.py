"""Filter rollout groups and fill grounding/gate quotas through generation rounds."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn

from .rewards import RouteCost, route_shaping

logger = logging.getLogger(__name__)

# The fields of a prompt as verl's RLHFDataset returns a row of avtr/rl/data.py's parquet (agent_name is
# not stored; the agent loop worker falls back to rollout.agent.default_agent_loop).
PROMPT_FIELDS = ("prompt", "raw_prompt", "data_source", "reward_model", "extra_info", "index", "tools_kwargs",
                 "interaction_kwargs", "dummy_tensor")
STD_FLOOR = 1e-8
SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class SamplingConfig:
    round_prompts: int
    max_rounds: int
    reroll_rounds: int
    grounding_groups: int
    seed: int
    grounding_weight: float
    cost: RouteCost

    @classmethod
    def from_stage(cls, stage) -> SamplingConfig:
        """From the ``avtr`` block of rl.yaml (``trainer.v1.sampler.sampler_kwargs``)."""
        s = stage["sampling"]
        cfg = cls(int(s["round_prompts"]), int(s["max_rounds"]), int(s["reroll_rounds"]), int(s["grounding_groups"]),
                  int(s["seed"]), float(stage["grounding_loss_weight"]),
                  RouteCost(float(stage["think_cost"]), float(stage["block_cost"]), float(stage["noblock_bonus"])))
        if cfg.round_prompts < 1 or cfg.max_rounds < 1 or cfg.reroll_rounds < 0 or cfg.grounding_groups < 0:
            raise ValueError(f"invalid avtr.sampling settings: {cfg}")
        return cfg


def total_rewards(rows: Sequence[dict], grounding_weight: float, cost: RouteCost) -> list[float]:
    """The reward of each rollout of one group (``rows`` are its agent-loop records): ``grounding_weight x
    IoU`` on a grounding group, accuracy plus the route shaping on a gate group."""
    info = [row["reward_extra_info"] for row in rows]
    scores = [float(x["score"]) for x in info]
    if rows[0]["task"] == "grounding":
        return [grounding_weight * s for s in scores]
    shaping = route_shaping(scores, [bool(x["fired"]) for x in info], [bool(x["thinking"]) for x in info],
                            [0] * len(rows), cost)
    return [s + r for s, r in zip(scores, shaping, strict=True)]


def classify(rewards: Sequence[Sequence[float]]) -> tuple[list[bool], list[bool]]:
    """``(qualified, uniformly_wrong)`` for the groups of one round (``rewards[i]``: group i's rollout rewards).
    A qualified group's rewards are not all equal (std at least ``STD_FLOOR``). A uniformly wrong group has no
    right rollout (no positive reward, or one low reward shared by all, such as a grounding group stuck on one
    poor window) and may be re-rolled. Computed on the round's (groups x rollouts) float32 matrix, as the reference
    does: torch's row-wise std of a constant row vanishes exactly, the std of a lone row need not."""
    if not rewards:
        return [], []
    r = torch.tensor(rewards, dtype=torch.float32)
    qualified = r.std(dim=1) >= STD_FLOOR
    wrong = (r <= 0).all(dim=1) | (~qualified & (r.amax(dim=1) <= 0.5))
    return qualified.tolist(), wrong.tolist()


def is_filled(n_grounding: int, n_gate: int, target: int, quota: int) -> bool:
    """Whether the qualified groups fill a step of ``target`` groups, ``quota`` of them grounding (unset: any mix)."""
    if quota <= 0:
        return n_grounding + n_gate >= target
    return n_grounding >= quota and n_gate >= target - quota


def draw(grounding: Sequence[bool], quota: int, target: int, seed: int) -> list[int]:
    """Sorted indices of the ``target`` groups a step keeps out of its qualified groups (``grounding[i]``: group
    i is a grounding group): ``quota`` grounding and the rest gate, each drawn uniformly, a task short of its
    share leaving the rest to the other; one uniform draw when no ``quota`` is set."""
    n = len(grounding)
    if n <= target:
        return list(range(n))
    gen = torch.Generator()
    gen.manual_seed(seed)
    if quota <= 0:
        return torch.randperm(n, generator=gen)[:target].sort().values.tolist()
    flags = torch.tensor(grounding, dtype=torch.bool)
    grounding_idx, gate_idx = flags.nonzero(as_tuple=True)[0], (~flags).nonzero(as_tuple=True)[0]
    n_gate = min(target - min(quota, len(grounding_idx)), len(gate_idx))
    n_grounding = min(target - n_gate, len(grounding_idx))
    take_grounding = grounding_idx[torch.randperm(len(grounding_idx), generator=gen)[:n_grounding]]
    take_gate = gate_idx[torch.randperm(len(gate_idx), generator=gen)[:n_gate]]
    return torch.cat([take_grounding, take_gate]).sort().values.tolist()


def _values(column) -> list:
    if isinstance(column, torch.Tensor):
        return list(column.unbind(0))
    values = column.tolist() if hasattr(column, "tolist") else list(column)
    return [getattr(v, "data", v) for v in values]


class QuotaSampler(ReplayBuffer):
    """verl's ``trainer.v1.sampler.custom_sampler``; ``sampler_kwargs`` is the ``avtr`` block of rl.yaml."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trainer_mode != "sync" or self.refill_fn is None:
            raise ValueError("avtr/rl/sampler.py needs verl's sync trainer and its refill_fn")
        self.cfg = SamplingConfig.from_stage(self.sampler_kwargs)

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        cfg = self.cfg
        if cfg.grounding_groups > batch_size:
            raise ValueError(f"avtr.sampling.grounding_groups {cfg.grounding_groups} > train_batch_size {batch_size}")
        kept: list[str] = []
        kept_grounding: list[bool] = []
        rounds = total = failed = 0
        retries_left, retry_rounds, retry_groups, before_retries = cfg.reroll_rounds, 0, 0, 0

        self._sync_metadata_from_transfer_queue()
        missing = cfg.round_prompts - len(self.prompt_global_steps[partition_id])
        if missing > 0:
            self.refill_fn(missing)
        while True:
            rounds += 1
            uids = self._finish_round(partition_id, set(kept))
            groups = self._read_groups(partition_id, uids)
            size = max(map(len, groups.values()), default=0)
            judged = [uid for uid in uids if uid in groups and len(groups[uid]) == size]
            qualified, uniformly_wrong = classify([total_rewards(groups[u], cfg.grounding_weight, cfg.cost)
                                                   for u in judged])
            total += len(uids)
            failed += len(uids) - len(judged)
            if retry_rounds and not before_retries:
                before_retries = len(kept)
            new = [uid for uid, q in zip(judged, qualified, strict=True) if q]
            wrong = [uid for uid, w in zip(judged, uniformly_wrong, strict=True) if w]
            kept += new
            kept_grounding += [groups[uid][0]["task"] == "grounding" for uid in new]
            n_grounding = sum(kept_grounding)
            logger.info("sampling round %d: kept %d/%d groups (total %d/%d, %d grounding)", rounds, len(new),
                        len(uids), len(kept), batch_size, n_grounding)
            done = (is_filled(n_grounding, len(kept) - n_grounding, batch_size, cfg.grounding_groups)
                    or rounds >= cfg.max_rounds)
            reroll = None
            if not done and retries_left > 0 and wrong:
                retries_left -= 1
                retry_rounds += 1
                retry_groups += len(wrong)
                reroll = self._prompt_batch(partition_id, wrong, global_steps)
            self._clear_groups(partition_id, set(uids) - set(new))
            if done:
                break
            if reroll is not None:
                self._submit(reroll)
            else:
                self.refill_fn(cfg.round_prompts)

        if len(kept) < batch_size:
            raise RuntimeError(f"dynamic sampling: {rounds} round(s) produced {len(kept)} groups with a reward spread "
                               f"out of {total}, short of train_batch_size {batch_size}; raise avtr.sampling."
                               "round_prompts or max_rounds")
        keep = draw(kept_grounding, cfg.grounding_groups, batch_size, cfg.seed + (global_steps - 1) * SEED_STRIDE)
        selected = [kept[i] for i in keep]
        self._clear_groups(partition_id, set(kept) - set(selected))
        batch = self._materialize_batch(partition_id, selected, dict(self.partitions[partition_id]))
        # The reference run's row order, which the trainer's seeded shuffle permutes: the groups as selected
        # (round, then dispatch order), each group's rollouts by session. TransferQueue lists them as they finished.
        place = {uid: i for i, uid in enumerate(selected)}

        def row_order(key: str) -> tuple[int, int, int]:
            uid, session, index = key.rsplit("_", 2)
            return place[uid], int(session), int(index)

        batch.reorder(sorted(range(len(batch.keys)), key=lambda j: row_order(batch.keys[j])))
        metrics = {
            "avtr/sampling/gen_rounds": rounds,
            "avtr/sampling/total_groups": total,
            "avtr/sampling/qualified_groups": len(kept),
            "avtr/sampling/surplus_groups": len(kept) - len(selected),
            "avtr/sampling/retry_rounds": retry_rounds,
            "avtr/sampling/retry_groups": retry_groups,
            "avtr/sampling/retry_rescued": max(0, len(kept) - before_retries) if retry_rounds else 0,
            "avtr/sampling/grounding_groups": sum(kept_grounding[i] for i in keep),
            "avtr/sampling/failed_groups": failed,
        }
        return batch, metrics

    def _finish_round(self, partition_id: str, kept: set[str]) -> list[str]:
        """The uids of the round in flight, in dispatch order, once every one of them has finished."""
        last_debug_time = time.time()
        while True:
            self._sync_metadata_from_transfer_queue()
            if not self.pending_keys[partition_id] and not self.running_keys[partition_id]:
                return [uid for uid in self.prompt_global_steps[partition_id] if uid not in kept]
            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

    def _read_groups(self, partition_id: str, uids: list[str]) -> dict[str, list[dict]]:
        """The agent-loop records (``extra_fields``) of each group of ``uids`` that finished with trajectories."""
        finished = {uid for uid in uids if uid in self.finished_keys[partition_id]}
        keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in finished]
        groups: dict[str, list[dict]] = {}
        if keys:
            data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["extra_fields"])
            for key, extra in zip(keys, _values(data["extra_fields"]), strict=True):
                groups.setdefault(key.split("_")[0], []).append(extra)
        return groups

    def _prompt_batch(self, partition_id: str, uids: list[str], global_steps: int):
        """The prompts of ``uids`` as a dataloader batch under new uids, read back from one trajectory each."""
        first: dict[str, str] = {}
        for key in self.partitions[partition_id]:
            first.setdefault(key.split("_")[0], key)
        data = tq.kv_batch_get(keys=[first[uid] for uid in uids], partition_id=partition_id,
                               select_fields=list(PROMPT_FIELDS))
        columns = {name: _values(data[name]) for name in data.keys()}
        batch = collate_fn([{name: values[i] for name, values in columns.items()} for i in range(len(uids))])
        batch["uid"] = np.array([str(uuid.uuid4()) for _ in uids], dtype=object)
        batch = tu.get_tensordict(batch)
        tu.assign_non_tensor_data(batch, "global_steps", global_steps)
        return batch

    def _submit(self, batch) -> None:
        """Dispatch ``batch`` through the trainer that ``refill_fn`` is bound to, as its fresh prompts are."""
        submit = getattr(getattr(self.refill_fn, "__self__", None), "_submit_batch_to_rollout", None)
        if submit is None:
            raise RuntimeError("a re-roll needs verl's trainer: refill_fn must be its bound _add_prompts_to_generate")
        submit(batch)
