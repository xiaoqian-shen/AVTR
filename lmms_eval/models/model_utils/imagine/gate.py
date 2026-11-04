"""The two gates of ``qwen3_5_imagine_vllm``: fire-or-decline at the trigger position, and the
optional reason-or-not probe right after ``<think>``."""

from __future__ import annotations

import math

from lmms_eval.models.model_utils.imagine.format import (
    FORCE_BIAS,
    MASK_BIAS,
)


class GateMixin:
    """Gate methods of ``Qwen3_5_ImagineVLLM``, whose ``__init__`` sets their state."""

    # ---- sampling params ------------------------------------------------------------------

    def _mask(self, ids) -> dict[int, float]:
        return {int(i): MASK_BIAS for i in ids}

    def _gate_params(self):
        """One token at the trigger position.

        Two modes, as in the former HF adapter. Unrestricted (default): the decision is the
        argmax, exactly as the former HF adapter computed it, and the reported margins come from the
        returned top-k -- absent when the marker is far out of contention, which the fire rate already tells
        you.
        Restricted (``imagine_prob_threshold`` set): a FORCE_BIAS on {OPEN, CLOSE, <think>} makes the
        drawn token the argmax over those three. The returned logprobs are still the RAW full-vocab
        top-k (vLLM gathers them before any bias), so the pairwise gate value is exact only when both
        of its tokens are in that top-k. A wide ``logprobs`` makes that the normal case; a row where one
        is missing has no gate value and declines, and is counted in ``_gate_missing``.
        """
        from vllm import SamplingParams

        if self._threshold is not None:
            return SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=20,
                logit_bias={
                    self._open_id: FORCE_BIAS,
                    self._close_id: FORCE_BIAS,
                    self._think_id: FORCE_BIAS,
                },
                skip_special_tokens=False,
            )
        return SamplingParams(
            temperature=0.0,  # the gate is a decision, never sampled -- as in the former HF adapter
            max_tokens=1,
            logprobs=20,
            logit_bias=self._mask(self._query_ids + self._tool_ids),
            skip_special_tokens=False,
        )

    # ---- gate -----------------------------------------------------------------------------

    def _gate_conf(self, comp) -> dict:
        """``sigmoid(logit_a - logit_b)`` for the two pairwise gates, from the returned logprobs.

        log-softmax differences equal logit differences, so this is exact whenever both tokens are
        present; a token outside the returned top-k reports None rather than a fabricated value.
        """
        if not comp.logprobs:
            return {}
        lp = {int(t): float(v.logprob) for t, v in comp.logprobs[0].items()}
        conf: dict[str, object] = {
            "argmax_id": int(comp.token_ids[0]),
            # In threshold mode the drawn token is forced into {OPEN, CLOSE, <think>}, so this is
            # the argmax over THREE tokens, not over the vocab. Recorded so a later reader cannot
            # mistake it for "the model would have fired on its own".
            "gate_restricted": self._threshold is not None,
            "fired_argmax": int(comp.token_ids[0]) == self._open_id,
            "p_open": (
                round(math.exp(lp[self._open_id]), 6) if self._open_id in lp else None
            ),
        }
        for key, other in (
            ("fire_vs_think", self._think_id),
            ("fire_vs_close", self._close_id),
        ):
            if self._open_id in lp and other in lp:
                gap = lp[self._open_id] - lp[other]
                conf[key] = round(1.0 / (1.0 + math.exp(-gap)), 6)
                conf[f"{key}_logit_gap"] = round(gap, 4)
            else:
                conf[key] = None
        return conf

    @staticmethod
    def _first_token_conf(comp) -> dict:
        """How sure the model was of the FIRST token it generated for the answer.

        Only meaningful on the NO-THINK arm, where that token is the answer itself (the rows read
        `<think>\n\n</think>\n\nA<|im_end|>`). On the reasoning arm the first token
        is chain text and this measures nothing about the answer -- which is fine, because the
        point of the signal is to let the CHEAP branch report its own uncertainty before anything
        decides to reason.
        """
        if not comp.logprobs or not comp.token_ids:
            return {}
        top = sorted(
            (float(v.logprob) for v in comp.logprobs[0].values()), reverse=True
        )
        ps = [math.exp(x) for x in top]
        drawn = comp.logprobs[0].get(int(comp.token_ids[0]))
        return {
            "p_top1": round(ps[0], 6),
            # The runner-up margin, which is what an MCQ answer's confidence actually is: p_top1
            # alone conflates "sure" with "the option set was small".
            "margin": round(ps[0] - ps[1], 6) if len(ps) > 1 else None,
            "entropy": round(-sum(q * math.log(max(q, 1e-12)) for q in ps), 6),
            "p_drawn": (
                round(math.exp(float(drawn.logprob)), 6) if drawn is not None else None
            ),
        }

    def _decide_think(self, conf: dict) -> bool:
        """Reason or not. argmax is a zero tau; an explicit threshold shifts that one operating point.

        A row whose probe returned nothing has no decision to make and must not silently become
        "reason" -- that is how a broken probe turns into an always-think rate that looks like a
        result.
        """
        gap = conf.get("reason_vs_empty_logit_gap")
        if gap is None:
            return bool(conf.get("argmax_reason"))
        if self._think_threshold is None:
            return bool(conf.get("argmax_reason"))
        return float(gap) > self._think_threshold

    def _decide(self, conf: dict) -> bool:
        """Fire or decline. The forced routes never consult the model: on a decline the peeked
        token is discarded, so a forced route has nothing to peek for."""
        if self._force_block:
            return True
        if self._mute:
            return False
        if self._threshold is not None:
            value = conf.get(self._gate_key)
            if value is None:
                self._gate_missing += 1
                return False
            return float(value) >= self._threshold
        return bool(conf.get("fired_argmax"))

    def _think_probe_params(self, only_id: int):
        """One token, FORCED to ``only_id``, so that id's raw logprob is always returned.

        Asking for ``allowed_token_ids=[nl, nl2]`` with a top-k as small as the pair does not
        renormalize over the pair: vLLM V1 computes the returned logprobs from the RAW logits
        (sampler.py, ``raw_logprobs = compute_logprobs(logits)``) and only then applies the
        allowed-ids mask, so that top-k is over the whole vocabulary. ``\\n`` is essentially never
        in it after ``<think>``, so such a probe throws away nearly every row -- and the rows it DOES
        keep are exactly the ones most inclined to reason, a selection perfectly correlated with the
        quantity being measured.

        Forcing one id makes it the sampled token, and the sampled token's raw logprob is always
        included. Two such requests give both terms of the gap. It is forced with FORCE_BIAS, not
        ``allowed_token_ids``, whose per-slot mask vLLM's batch can leave on another request.
        """
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=1,
            logit_bias={only_id: FORCE_BIAS},
            skip_special_tokens=False,
        )

    def _think_probe_single_params(self):
        """Both probe ids from ONE forward.

        The raw logprobs are gathered before any bias is applied (see ``_think_probe_params``), so
        a wide ``logprobs`` returns both ids whenever both are in the raw top-k -- on the checkpoint
        this harness was validated on, they are the two most likely tokens on every row. The SAMPLED
        token is never read as the decision: greedy sends an exact tie to the lower id, ``\\n``,
        while the rule sends it to no-think.
        """
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            logit_bias={self._nl_id: FORCE_BIAS, self._nl2_id: FORCE_BIAS},
            skip_special_tokens=False,
        )

    @staticmethod
    def _raw_lps(comp) -> dict[int, float]:
        if not comp.logprobs:
            return {}
        return {int(t): float(v.logprob) for t, v in comp.logprobs[0].items()}

    @staticmethod
    def _think_conf_from_lps(a: float | None, b: float | None) -> dict:
        """think_conf from the raw lp(\\n) and lp(\\n\\n) at the position right after ``<think>``."""
        if a is None or b is None:
            return {}
        gap = a - b
        return {
            "reason_vs_empty": round(1.0 / (1.0 + math.exp(-gap)), 6),
            "reason_vs_empty_logit_gap": round(gap, 4),
            "argmax_reason": gap > 0.0,
            # The two RAW log probs, not just their difference. The gap alone cannot distinguish a
            # question the model has an opinion about from one where it wants some third token
            # entirely and is being forced to pick between two it does not care about -- those
            # have identical gaps and very different meanings. Their sum also gives the pair's
            # share of the full-vocabulary mass, which is the think-gate counterpart of the visual
            # gate's p_open / fire_vs_close.
            "lp_think": round(a, 4),
            "lp_nothink": round(b, 4),
            "pair_mass": round(math.exp(a) + math.exp(b), 6),
        }
