"""
Keep briefly-undetected players on the minimap at their last known position.

Detections flicker (occlusion by the net or other players, motion blur). Instead of
the dot vanishing, a player missing for up to `max_hold_frames` is reported with
`observed=False` so it can be drawn semi-transparent.
"""
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from court_mapping.projection import PlayerPosition
from court_mapping.smoothing import TrackPoint

DEFAULT_SUPPRESS_RADIUS_M = 0.8


@dataclass(frozen=True)
class HoldState:
    last_seen: Mapping[int, TrackPoint] = field(default_factory=lambda: MappingProxyType({}))


def hold_missing(state: HoldState,
                 observed: Iterable[PlayerPosition],
                 frame_index: int,
                 max_hold_frames: int,
                 suppress_radius_m: float = DEFAULT_SUPPRESS_RADIUS_M
                 ) -> tuple[HoldState, tuple[PlayerPosition, ...]]:
    """
    Return the new state and the held (unobserved) players for this frame.

    A held player is dropped when its hold expires, or when another observed player
    stands within `suppress_radius_m` — almost always the same person under a new
    tracker ID, which would otherwise show up twice.
    """
    current = tuple(observed)
    current_ids = {p.tracker_id for p in current}

    def still_held(tracker_id: int, point: TrackPoint) -> bool:
        return (tracker_id not in current_ids
                and frame_index - point.last_frame <= max_hold_frames
                and not _near_any(point, current, suppress_radius_m))

    kept = {tid: pt for tid, pt in state.last_seen.items() if still_held(tid, pt)}
    held = tuple(PlayerPosition(tid, pt.x_m, pt.y_m, observed=False) for tid, pt in kept.items())
    seen_now = {p.tracker_id: TrackPoint(p.x_m, p.y_m, frame_index) for p in current}
    return HoldState(MappingProxyType({**kept, **seen_now})), held


def _near_any(point: TrackPoint, players: tuple[PlayerPosition, ...], radius_m: float) -> bool:
    return any(math.hypot(p.x_m - point.x_m, p.y_m - point.y_m) < radius_m for p in players)
