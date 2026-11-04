
"""Generation settings and answer extraction shared by the Qwen3.5 adapters."""

from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

_T = TypeVar("_T")

THINK_MAX_NEW_TOKENS_FLOOR = 8192
NO_THINK_MAX_NEW_TOKENS_FLOOR = 32
SC_TOP_P = 0.95
SC_TOP_K = 20

_THINKING_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 32768,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
}
_NO_THINKING_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
}
_MERGE_KEYS = ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty")
_FINAL_ANSWER_PATTERNS = (
    r"<answer>\s*(.*?)\s*</answer>",
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
)


def resolve_attn_implementation(*, fallback: bool) -> str:
    """The ``attn_implementation`` to request for an FA2 load.

    Without ``fallback`` it is always ``flash_attention_2``, so ``from_pretrained`` raises on a
    build without the kernel (lean local builds). With it, such a build gets ``sdpa``, which is
    functionally equivalent but not bit-identical.
    """
    if not fallback:
        return "flash_attention_2"
    try:
        importlib.import_module("flash_attn")
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def _group_by_gen_kwargs(gen_kwargs: Sequence[object]) -> list[list[int]]:
    """Indices grouped by equal ``gen_kwargs``, groups in first-seen order.

    Compared with ``==`` rather than hashed, because the values hold YAML lists and dicts.
    """
    groups: dict[int, list[int]] = {}
    for i, kwargs in enumerate(gen_kwargs):
        first = next((g for g in groups if gen_kwargs[g] == kwargs), i)
        groups.setdefault(first, []).append(i)
    return list(groups.values())


def run_per_gen_kwargs(
    args: Sequence[tuple[Any, ...]],
    run: Callable[[list[tuple[Any, ...]]], Sequence[_T]],
) -> list[_T]:
    """Calls ``run`` once per group of request ``args`` with equal ``generation_kwargs`` (``a[1]``
    of each request ``a``), then puts the answers back in request order.

    The evaluator sends every task's ``generate_until`` requests in one call, so one call can
    carry several budgets, stop lists and temperatures. A call with a single config runs ``run``
    once, on the requests in their original order.
    """
    answers: dict[int, _T] = {}
    for idx in _group_by_gen_kwargs([a[1] for a in args]):
        out = run([args[i] for i in idx])
        if len(out) != len(idx):
            raise RuntimeError(
                f"generation returned {len(out)} answers for {len(idx)} requests"
            )
        answers.update(zip(idx, out))
    return [answers[i] for i in range(len(args))]


def resolve_gen_kwargs(
    gen_kwargs: Mapping[str, Any],
    *,
    enable_thinking: bool,
    floor_max_new_tokens: bool = False,
    min_new_tokens: int | None = None,
    sample_temperature: float | None = None,
) -> dict[str, Any]:
    """Sampling settings for one request group.

    The mode's defaults, overlaid by the task's ``generation_kwargs``. The result always carries
    ``min_new_tokens``; ``None`` (the default) means ``generate()`` must not be given one, and the
    task's own ``min_new_tokens`` is then ignored as well. Any int opts in, with the task's value
    taking precedence over the model-level one.
    """
    current = dict(_THINKING_DEFAULTS if enable_thinking else _NO_THINKING_DEFAULTS)
    # Merge user-supplied kwargs over defaults so callers can override
    # max_new_tokens/temperature/etc., as the upstream Qwen-VL adapters do.
    for key in _MERGE_KEYS:
        if key in gen_kwargs:
            current[key] = gen_kwargs[key]

    current["min_new_tokens"] = None
    if min_new_tokens is not None:
        current["min_new_tokens"] = int(
            gen_kwargs.get("min_new_tokens", min_new_tokens) or 0
        )

    if floor_max_new_tokens:
        # Thinking: the short budgets of tasks like mvbench, blink or mmsi truncate the CoT before
        # </think>, so no final-answer wrapper is ever produced. The thinking floor lets a short CoT
        # close; a much larger one lets a few runaway decodes dominate the run, and many thinking-on
        # tasks did not finish. No-think: a tiny budget fits "C" but not "The correct answer is C",
        # and prose-heavy prompts (e.g. videomme_w_subtitle) do drift into a sentence. The floor is
        # applied whether or not min_new_tokens is set.
        floor = (
            THINK_MAX_NEW_TOKENS_FLOOR
            if enable_thinking
            else NO_THINK_MAX_NEW_TOKENS_FLOOR
        )
        current["max_new_tokens"] = max(int(current["max_new_tokens"]), floor)

    if sample_temperature is not None:
        current["temperature"] = sample_temperature

    if current["temperature"] > 0:
        current["do_sample"] = True
    else:
        current["do_sample"] = False
        current["temperature"] = None
        current["top_p"] = None
        current["top_k"] = None
    return current


def self_consistency_kwargs(
    gen_kwargs: Mapping[str, Any], k: int, sc_temperature: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(gen_kwargs, extra generate() kwargs)`` for ``k`` samples per prompt.

    ``k <= 1`` changes nothing. Otherwise HF expands the batch prompt-major (row ``i * k + j`` is
    sample ``j`` of prompt ``i``), and greedy decoding becomes sampling, since ``k`` greedy samples
    of one prompt would be identical.
    """
    if k <= 1:
        return dict(gen_kwargs), {}
    extra: dict[str, Any] = {"num_return_sequences": k}
    if gen_kwargs["do_sample"]:
        return dict(gen_kwargs), extra
    sampled: dict[str, Any] = {
        **gen_kwargs,
        "do_sample": True,
        "temperature": sc_temperature,
        "top_p": SC_TOP_P,
        "top_k": SC_TOP_K,
    }
    return sampled, extra


def majority_vote(cands: Sequence[str]) -> str:
    """The most common candidate; a tie goes to the tied candidate seen first."""
    return Counter(cands).most_common(1)[0][0]


def apply_bare_think(
    texts: Sequence[str], bare_think: bool, enable_thinking: bool
) -> list[str]:
    """End thinking-mode prompts at ``<think>`` instead of the template's ``<think>\\n``.

    Tokenized as one piece, a no-think turn is ``<think>`` + ``\\n\\n`` + ``</think>`` while a
    reasoning turn is ``<think>`` + ``\\n`` + content, so the bare opener leaves both branches
    in-distribution. After the template's ``<think>\\n`` the no-think exit needs a second
    standalone ``\\n`` the base model never emits there.
    """
    if not (bare_think and enable_thinking):
        return list(texts)
    return [t[:-1] if t.endswith("<think>\n") else t for t in texts]


def has_reasoning(raw: str) -> bool:
    """Whether ``raw`` closed a NON-EMPTY think block.

    The opening ``<think>`` is usually part of the prompt rather than the generation, so only the
    text before the first ``</think>`` is inspected.
    """
    head, closed, _ = raw.partition("</think>")
    return bool(closed and head.replace("<think>", "").strip())


def extract_final_answer(text: str) -> str:
    """Content of the LAST ``<answer>...</answer>`` or ``\\boxed{...}`` in ``text``, stripped.

    Last rather than first, so a wrapper quoted inside the reasoning cannot shadow the final one.
    Text without either wrapper is returned unchanged for the task's own extractor.
    """
    best_end = -1
    best: str | None = None
    for pattern in _FINAL_ANSWER_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            if match.end() >= best_end:
                best_end, best = match.end(), match.group(1)
    return text if best is None else best.strip()


def postprocess_answer(
    ans: str,
    until: Sequence[str],
    *,
    enable_thinking: bool,
    extract_final: bool = False,
) -> str:
    """What the task scores: cut at the stop strings, drop a closed think block, then
    optionally reduce to the final ``<answer>``/``\\boxed`` content."""
    for term in until:
        if len(term) > 0:
            ans = ans.split(term)[0]
    if enable_thinking:
        _, think_end, remaining = ans.partition("</think>")
        # If the closing tag is missing (e.g. truncation), fall back to
        # the original answer instead of silently returning empty string.
        if think_end:
            ans = remaining.strip()
    if extract_final:
        ans = extract_final_answer(ans)
    return ans
