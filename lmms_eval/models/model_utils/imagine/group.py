"""What one group of ``qwen3_5_imagine_vllm`` rows produces, and the encoder tally it reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger as eval_logger


@dataclass
class GroupResult:
    """One entry per row of the group in every per-row list.

    ``gen`` is everything generated after the prompt, the injected block and scaffold included;
    ``ngen`` counts only the generated tokens. ``forced`` holds the rows whose chain was
    force-closed at ``think_budget``. The group is one engine loop, so ``wall_s`` is its wall clock
    and the per-row timeline is in ``extra``.
    """

    gen: list[str]
    ngen: list[int]
    blocks: list[int]
    trunc: list[bool]
    fires: list[bool]
    confs: list[dict]
    t_confs: list[dict]
    thinks: list[bool] | None
    forced: set[int]
    answer_conf: dict[int, dict]
    wall_s: float
    steps: int
    # Media items the encoder ran for this group.
    encodes: int
    extra: list[dict] = field(default_factory=list)


class EncodeTallyMixin:
    """Media items the encoder ran, per request id, and requests preempted, counted on vLLM's
    scheduler.

    A follow-up that re-runs the vision tower is otherwise invisible: RequestOutput reports only the
    KV prefix hit, and vLLM exposes encoder use only as an aggregate. A preemption frees a running
    request's KV, which it then prefills again.
    """

    def _install_encode_tally(self) -> None:
        self._encodes: dict[str, int] = {}
        self._preemptions: int = 0
        try:
            sched = self._llm.llm_engine.engine_core.engine_core.scheduler
            schedule = sched.schedule
            preempt = sched._preempt_request
        except Exception as e:  # noqa: BLE001 -- a private path must never fail the eval
            eval_logger.warning("imagine_vllm: encode tally not installed: {}", e)
            return
        tally = self._encodes

        def counted_preempt(*args, **kwargs):
            self._preemptions += 1
            return preempt(*args, **kwargs)

        def counted(*args, **kwargs):
            out = schedule(*args, **kwargs)
            for rid, items in (
                getattr(out, "scheduled_encoder_inputs", None) or {}
            ).items():
                # vLLM's internal id is the external one plus "-<random suffix>".
                key = rid.rsplit("-", 1)[0]
                tally[key] = tally.get(key, 0) + len(items)
            return out

        sched.schedule = counted
        sched._preempt_request = counted_preempt

    def _take_encodes(self, rid: str) -> int:
        """Encodes counted for ``rid``, forgetting them."""
        return self._encodes.pop(rid, 0)
