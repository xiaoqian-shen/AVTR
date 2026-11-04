"""Answer correctness, route costs, and temporal IoU rewards."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------------------------
# The answer must be a terminal commitment, not a letter that appears somewhere: a bare [A-H] scan
# over a reasoning chain reads the options the chain names while it deliberates.
_BOXED = re.compile(r"\\boxed\{\s*\(?([A-Ha-h])\)?\b")
_SOLE = re.compile(r"^\**\(?([A-Ha-h])\)?[.):\]]?\**$")
_SAID = re.compile(r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)\s*\**\(?([A-Ha-h])\)?\b", re.IGNORECASE)
_LETTER_GOLD = re.compile(r"^\(?([A-H])[.)\]]?\.?$", re.I)
_SPECIAL_TAIL = re.compile(r"(?:<\|[^>]{0,32}\|>|</?s>)+\s*$")
_MARKER = re.compile(r"\A\**\(?([A-Ha-h])\)?[.):\]]?\**[\s,:-]+(?:point|option|choice|view)\b", re.I)
_OPTION_LETTER = re.compile(r"\b[A-H]\b")
_ARTICLE_A = re.compile(r"(?:\A|[.!?:]\s+|\n\s*)\**A(?= [a-z])")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_NUM_GOLD = re.compile(r"^(-?[\d.]+)\s*(?:m|cm|meters?|centimeters?)?\.?$", re.I)
_ANSWER_IS = re.compile(r"^\s*(the\s+)?(answer|it)\s+(is|are)\s*[:=]?\s*")


def final_segment(response: str) -> str:
    """The text after the last ``</think>``; without one, the last non-empty line (a truncated chain's
    last line is prose, which matches nothing, so it cannot donate a letter it merely mentioned)."""
    text = (response or "").strip()
    if not text:
        return ""
    _, sep, tail = text.rpartition("</think>")
    if not sep:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        tail = lines[-1] if lines else ""
    return _SPECIAL_TAIL.sub("", tail.strip()).strip()


def final_letter(response: str) -> str | None:
    """The option letter the response commits to, or None."""
    segment = final_segment(response)
    if not segment:
        return None
    m = _BOXED.search(segment) or _SOLE.match(segment)
    if m:
        return m.group(1).upper()
    said = _SAID.findall(segment)
    if said:
        return said[-1].upper()
    if "</think>" not in (response or ""):
        return None
    # A closed answer may restate the option ("D. The camera moves closer"); it counts when it names
    # exactly one option letter (an "A" opening a sentence before a lowercase word is the article).
    m = _MARKER.match(segment)
    if m:
        return m.group(1).upper()
    articles = {m.end() - 1 for m in _ARTICLE_A.finditer(segment)}
    letters = {m.group(0) for m in _OPTION_LETTER.finditer(segment) if m.start() not in articles}
    return letters.pop() if len(letters) == 1 else None


def _plain(text: str) -> str:
    text = _ANSWER_IS.sub("", text.lower()).strip().strip(".!?,:;\"'()[]*")
    return re.sub(r"\s+", " ", text).strip()


def answer_score(response: str, gold: str) -> float:
    """Full credit if the final answer matches ``gold``, none otherwise. ``gold`` is an option letter, a number
    (within a relative tolerance with an absolute floor), or a word, which the whole answer must be (after
    lowercasing and stripping punctuation and a leading "the answer is" / "it is")."""
    gold = str(gold or "").strip()
    letter = _LETTER_GOLD.match(gold)
    if letter:
        return float(final_letter(response) == letter.group(1).upper())
    segment = final_segment(response)
    m = _NUM_GOLD.match(gold)
    if m is None:
        return float(bool(gold) and _plain(segment) == _plain(gold))
    nums = _NUM.findall(segment)
    if not nums:
        return 0.0
    try:
        target = float(m.group(1))
        return float(abs(float(nums[-1]) - target) <= max(0.1 * abs(target), 0.05))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------------------------
# Route cost
# ---------------------------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_route(response: str) -> tuple[bool, bool, int]:
    """``(fired, closed_think, think_chars)``. A decline carries only the bare ``</imagine>``, so
    firing needs the open marker."""
    fired = "<imagine>" in response
    m = _THINK_RE.search(response)
    if m is not None:
        return fired, True, len(m.group(1).strip())
    close = response.find("</think>")
    if close == -1:
        start = response.find("<think>")
        return (fired, False, 0) if start == -1 else (fired, False, len(response[start + len("<think>"):].strip()))
    return fired, True, len(response[:close].strip())


@dataclass(frozen=True)
class RouteCost:
    """Charged to CORRECT rollouts only, so an all-wrong group stays flat (no route preference is
    invented from noise) and a correct rollout always outranks a wrong one."""

    think_cost: float = 0.04  # per reasoning chain
    block_cost: float = 0.01  # per fired block
    noblock_bonus: float = 0.01  # to a non-firing rollout where firing and declining both solved it
    unclosed_min_chars: int = 200  # an unclosed chain counts as reasoning above this many characters

    def thinking(self, closed: bool, chars: int) -> bool:
        """Whether a parsed chain counts as reasoning. An unclosed chain also holds the answer, so it
        needs more than an answer's length."""
        return chars > (0 if closed else self.unclosed_min_chars)


def route_shaping(accs: Sequence[float], fired: Sequence[bool], thinking: Sequence[bool], groups: Sequence[object],
                  cost: RouteCost) -> list[float]:
    """Per-row route reward. ``groups[i]`` is row i's prompt key: the bonus depends on what the other
    rollouts of the same prompt did."""
    members: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    both_ok = {g: any(accs[i] > 0.5 for i in idx if fired[i]) and any(accs[i] > 0.5 for i in idx if not fired[i])
               for g, idx in members.items()}
    scores = []
    for i, g in enumerate(groups):
        if accs[i] <= 0.5:
            scores.append(0.0)
            continue
        s = cost.noblock_bonus if both_ok[g] and not fired[i] else 0.0
        scores.append(s - cost.think_cost * thinking[i] - cost.block_cost * fired[i])
    return scores


# ---------------------------------------------------------------------------------------------
# Temporal grounding
# ---------------------------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_T_RE = re.compile(r"<t>\s*(\d+(?:\.\d+)?)\s*</t>")
_BRACKET_RE = re.compile(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*[,;]\s*([-+]?\d+(?:\.\d+)?)\s*\]")
_SPAN_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
# The injected block is not model output, and its query indices are digits: a positional number scan
# would read every firing rollout as the span of its first query indices.
_BLOCK_RE = re.compile(r"</?imagine>|<latent_query_\d+>")


def _ordered(a: float, b: float) -> tuple[float, float]:
    return (a, b) if a <= b else (b, a)


def parse_span(text: str | None) -> tuple[float, float] | None:
    """``From <t>a</t>s to <t>b</t>s`` (first and last tag, optionally inside ``<answer>``)."""
    body = (text or "").strip()
    m = _ANSWER_RE.search(body)
    tags = _T_RE.findall(m.group(1).strip() if m else body)
    return _ordered(float(tags[0]), float(tags[-1])) if len(tags) >= 2 else None


def parse_span_loose(text: str | None) -> tuple[float, float] | None:
    """The predicted window: ``<t>`` tags, else the LAST ``[a, b]`` (a deliberating answer lists
    candidates before it commits), else the first pair of numbers, all after the block is stripped."""
    span = parse_span(text)
    if span is not None:
        return span
    body = _BLOCK_RE.sub(" ", text or "")
    brackets = _BRACKET_RE.findall(body)
    if brackets:
        return _ordered(float(brackets[-1][0]), float(brackets[-1][1]))
    nums = _SPAN_NUM_RE.findall(body)
    return _ordered(float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def gold_windows(gold) -> list[tuple[float, float]]:
    """Gold windows from ``[[a, b], ...]`` or ``<t>`` text (one tag pair per window)."""
    if not isinstance(gold, str):
        return [_ordered(float(a), float(b)) for a, b in gold]
    m = _ANSWER_RE.search(gold)
    tags = [float(t) for t in _T_RE.findall(m.group(1).strip() if m else gold)]
    if len(tags) <= 2 or len(tags) % 2:
        span = parse_span(gold)
        return [span] if span is not None else []
    return [_ordered(a, b) for a, b in zip(tags[0::2], tags[1::2], strict=True)]


def span_iou(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0.0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0.0 else 0.0


def tg_iou(response: str, gold) -> float:
    """IoU of the predicted window with the best-matching gold window (plain IoU, no GIoU)."""
    pred = parse_span_loose(response.rpartition("</think>")[2] if "</think>" in response else response)
    return max((span_iou(pred, w) for w in gold_windows(gold)), default=0.0)
