"""Turn per-window game-state predictions into clip segments (pure functions)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

INACTIVE_STATES = frozenset({"no-play", "unknown"})


@dataclass(frozen=True)
class Window:
    """One classification window: frames [start_frame, end_frame)."""
    start_frame: int
    end_frame: int
    state: str
    confidence: float


@dataclass(frozen=True)
class Segment:
    """A clip to export: frames [start_frame, end_frame)."""
    start_frame: int
    end_frame: int
    fps: float
    states: tuple[str, ...]

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


@dataclass(frozen=True)
class SegmentConfig:
    pad_before_s: float = 1.0      # extra context before each segment
    pad_after_s: float = 1.0       # extra context after each segment
    merge_gap_s: float = 1.5       # join segments separated by a shorter no-play gap
    min_duration_s: float = 1.0    # drop segments shorter than this (before padding)
    smooth_k: int = 3              # majority filter width over windows (odd, 1 = off)


def smooth_states(states: Sequence[str], k: int) -> list[str]:
    """Majority filter of width k; ties keep the original state."""
    if k < 1 or k % 2 == 0:
        raise ValueError(f"smooth_k must be a positive odd number, got {k}")
    if k == 1:
        return list(states)
    half = k // 2
    result = []
    for i, current in enumerate(states):
        votes = Counter(states[max(0, i - half): i + half + 1])
        top, top_count = votes.most_common(1)[0]
        result.append(current if votes[current] == top_count else top)
    return result


def _active_runs(windows: Sequence[Window], states: Sequence[str]) -> list[tuple[int, int, tuple[str, ...]]]:
    runs: list[tuple[int, int, tuple[str, ...]]] = []
    for window, state in zip(windows, states):
        if state in INACTIVE_STATES:
            continue
        if runs and runs[-1][1] >= window.start_frame:
            start, _, seen = runs[-1]
            new_seen = seen if state in seen else seen + (state,)
            runs[-1] = (start, window.end_frame, new_seen)
        else:
            runs.append((window.start_frame, window.end_frame, (state,)))
    return runs


def _merge(runs: list[tuple[int, int, tuple[str, ...]]], max_gap: int) -> list[tuple[int, int, tuple[str, ...]]]:
    merged: list[tuple[int, int, tuple[str, ...]]] = []
    for start, end, states in runs:
        if merged and start - merged[-1][1] <= max_gap:
            p_start, p_end, p_states = merged[-1]
            extra = tuple(s for s in states if s not in p_states)
            merged[-1] = (p_start, max(p_end, end), p_states + extra)
        else:
            merged.append((start, end, states))
    return merged


def build_segments(windows: Sequence[Window], fps: float, total_frames: int,
                   config: SegmentConfig) -> list[Segment]:
    """Smooth → collect active runs → merge small gaps → drop short → pad → merge overlaps."""
    if not windows:
        return []
    states = smooth_states([w.state for w in windows], config.smooth_k)
    runs = _merge(_active_runs(windows, states), round(config.merge_gap_s * fps))
    min_len = config.min_duration_s * fps
    runs = [r for r in runs if r[1] - r[0] >= min_len]

    pad_before, pad_after = round(config.pad_before_s * fps), round(config.pad_after_s * fps)
    padded = [(max(0, s - pad_before), min(total_frames, e + pad_after), st) for s, e, st in runs]
    return [Segment(s, e, fps, st) for s, e, st in _merge(padded, 0)]
