"""Game-state segment model + validation of a timeline before it is saved.

Frames are inclusive on both ends, exactly like the `game_state_segments` table
written by the original Qt annotator (vb_annotator). The editing logic itself
(layers, hold-to-label, compose) lives in the browser (`static/js/ops.js`);
the server only validates and stores what the user saves with Ctrl+S.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

STATES = ("service", "play", "no-play")
MANUAL = "manual"
MODEL = "model"
SOURCES = (MANUAL, MODEL)

Timeline = tuple["Segment", ...]


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    state: str
    source: str = MANUAL
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"unknown state {self.state!r}; expected one of {STATES}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(f"invalid frame range [{self.start_frame}, {self.end_frame}]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {self.confidence}")


CONFIDENCE_DIGITS = 4


def _length(seg: Segment) -> int:
    return seg.end_frame - seg.start_frame + 1


def _merge_pair(a: Segment, b: Segment) -> Segment:
    """One segment for two touching same-state segments: manual wins, confidence is length-weighted."""
    source = MANUAL if MANUAL in (a.source, b.source) else MODEL
    confidence = (a.confidence * _length(a) + b.confidence * _length(b)) / (_length(a) + _length(b))
    return replace(a, end_frame=b.end_frame, source=source, confidence=confidence)


def normalize(segments) -> Timeline:
    """Sort, reject overlaps, and merge every run of touching segments that share a state.

    A merged segment is "manual" if any part was manual (manual overrides model);
    its confidence is the length-weighted mean of its parts. Same rule as the
    browser's `mergeSameState` in static/js/ops.js.
    """
    ordered = sorted(segments, key=lambda s: (s.start_frame, s.end_frame))
    merged: list[Segment] = []
    for seg in ordered:
        prev = merged[-1] if merged else None
        if prev is not None and seg.start_frame <= prev.end_frame:
            raise ValueError(f"片段重疊：{prev} 與 {seg}")
        if prev is not None and prev.end_frame + 1 == seg.start_frame and prev.state == seg.state:
            merged[-1] = _merge_pair(prev, seg)
        else:
            merged.append(seg)
    return tuple(replace(s, confidence=round(s.confidence, CONFIDENCE_DIGITS)) for s in merged)


def validate_for_video(segments, frame_count: int) -> Timeline:
    """Normalize and make sure every segment lies inside the video."""
    timeline = normalize(segments)
    if timeline and timeline[-1].end_frame >= frame_count:
        raise ValueError(f"片段超出影片範圍：最後一幀是 {frame_count - 1}，收到 {timeline[-1].end_frame}")
    return timeline
