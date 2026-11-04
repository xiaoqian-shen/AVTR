
"""First-occurrence dedup for per-request work that is really per-DOC work."""

from __future__ import annotations

from typing import Hashable, Sequence


def dedup_slots(keys: Sequence[Hashable]) -> tuple[list[int], list[int]]:
    """``(first_occurrence_indices, slot_per_input)`` for ``keys``.

    ``slot_per_input[i]`` indexes into a list built from ``first_occurrence_indices``, so a caller
    can compute an expensive value once per distinct key and then fan it back out in the original
    order -- including when equal keys are interleaved rather than adjacent.
    """
    first: dict[Hashable, int] = {}
    uniq: list[int] = []
    slots: list[int] = []
    for i, key in enumerate(keys):
        if key not in first:
            first[key] = len(uniq)
            uniq.append(i)
        slots.append(first[key])
    return uniq, slots
