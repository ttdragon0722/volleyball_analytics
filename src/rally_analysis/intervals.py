"""Frame-interval arithmetic: which parts of a reviewed range are NOT no-play.

All spans are half-open [start, end) in frame numbers.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Span = tuple[int, int]
UNLABELLED = "unlabelled"


@dataclass(frozen=True)
class LabeledSpan:
    start: int
    end: int
    state: str


def keep_spans(reviewed: Span, excluded: Iterable[Span], min_len: int) -> tuple[Span, ...]:
    """`reviewed` minus every `excluded` span; pieces shorter than `min_len` are dropped."""
    lo, hi = reviewed
    clipped = sorted((max(s, lo), min(e, hi)) for s, e in excluded if e > lo and s < hi)
    pieces: list[Span] = []
    cursor = lo
    for s, e in clipped:
        if s > cursor:
            pieces.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        pieces.append((cursor, hi))
    return tuple(p for p in pieces if p[1] - p[0] >= min_len)


def states_in(span: Span, labels: Iterable[LabeledSpan]) -> tuple[str, ...]:
    """Distinct states overlapping `span`, in order of first appearance."""
    lo, hi = span
    ordered = sorted((l for l in labels if l.end > lo and l.start < hi), key=lambda l: l.start)
    states = tuple(dict.fromkeys(l.state for l in ordered))
    return states or (UNLABELLED,)
