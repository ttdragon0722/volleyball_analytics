"""Per-tracker exponential smoothing of court positions (immutable state)."""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from court_mapping.projection import PlayerPosition

DEFAULT_ALPHA = 0.4
DEFAULT_MAX_GAP_FRAMES = 15


@dataclass(frozen=True)
class TrackPoint:
    x_m: float
    y_m: float
    last_frame: int


@dataclass(frozen=True)
class SmootherState:
    tracks: Mapping[int, TrackPoint] = field(default_factory=lambda: MappingProxyType({}))


def smooth_positions(state: SmootherState,
                     positions: Iterable[PlayerPosition],
                     frame_index: int,
                     alpha: float = DEFAULT_ALPHA,
                     max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES
                     ) -> tuple[SmootherState, tuple[PlayerPosition, ...]]:
    """
    Blend each new position with that tracker's previous smoothed position.
    alpha = weight of the new observation (1.0 = no smoothing).
    Tracks unseen for more than `max_gap_frames` are reset / pruned.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    def is_fresh(track: TrackPoint) -> bool:
        return frame_index - track.last_frame <= max_gap_frames

    live = {tid: t for tid, t in state.tracks.items() if is_fresh(t)}
    smoothed = tuple(_blend(live.get(p.tracker_id), p, alpha) for p in positions)
    updated = {**live, **{p.tracker_id: TrackPoint(p.x_m, p.y_m, frame_index) for p in smoothed}}
    return SmootherState(MappingProxyType(updated)), smoothed


def _blend(previous: TrackPoint | None, current: PlayerPosition, alpha: float) -> PlayerPosition:
    if previous is None:
        return current
    return replace(
        current,
        x_m=alpha * current.x_m + (1 - alpha) * previous.x_m,
        y_m=alpha * current.y_m + (1 - alpha) * previous.y_m,
    )
