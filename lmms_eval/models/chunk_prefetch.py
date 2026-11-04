
"""Background preparation of a stream of chunks, for a consumer that generates on one chunk while
the next ones are decoded."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def split_chunks(
    items: Sequence[T], cap: int, first: int | None = None
) -> list[Sequence[T]]:
    """``items`` in chunks of ``cap``; with ``first``, the chunks start at ``first`` items and
    double up to ``cap``."""
    size = max(1, min(first or cap, cap))
    chunks, i = [], 0
    while i < len(items):
        chunks.append(items[i : i + size])
        i += size
        size = min(2 * size, cap)
    return chunks


def prefetch(
    chunks: Sequence[T],
    prepare: Callable[[T], R],
    ahead: int = 1,
    on_ready: Callable[[int, float], None] | None = None,
) -> Iterator[R]:
    """Yield ``prepare(chunk)`` per chunk, in order. One background thread prepares the chunks one
    after another, staying up to ``ahead`` chunks ahead of the consumer. ``on_ready(index,
    waited_s)`` runs as each chunk is handed over, with the time the consumer spent blocked on it.
    """
    ahead = max(1, ahead)
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = deque(pool.submit(prepare, c) for c in chunks[:ahead])
        for ci in range(len(chunks)):
            t0 = time.perf_counter()
            result = futures.popleft().result()
            waited = time.perf_counter() - t0
            if ci + ahead < len(chunks):
                futures.append(pool.submit(prepare, chunks[ci + ahead]))
            if on_ready is not None:
                on_ready(ci, waited)
            yield result
