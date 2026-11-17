"""
Action-aware correction for jumping players.

While a player is in the air the bottom of their box is not on the floor, so the
homography pushes them away from the camera (a blocker can land 4 m inside the
opponent's half). Jump actions (block, spike, serve, set) are vertical or nearly so,
so the player's position is frozen at the take-off point until shortly after landing.

The action detector only fires for part of the jump, so:
- the anchor is taken `lookback_frames` before the detection (take-off, not mid-air);
- the freeze lasts `release_frames` after the last detection (hang time + landing).
"""
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from court_mapping.projection import PlayerPosition
from court_mapping.smoothing import TrackPoint

DEFAULT_JUMP_ACTIONS = frozenset({"block", "spike", "serve", "set"})
DEFAULT_MIN_IOU = 0.2  # set boxes overlap the pose box loosely (~0.22 IoU observed)
DEFAULT_LOOKBACK_S = 0.25
DEFAULT_RELEASE_S = 0.5


@dataclass(frozen=True)
class AirborneConfig:
    jump_actions: frozenset[str] = DEFAULT_JUMP_ACTIONS
    min_iou: float = DEFAULT_MIN_IOU
    lookback_frames: int = 8
    release_frames: int = 15

    @classmethod
    def from_seconds(cls, fps: float, lookback_s: float = DEFAULT_LOOKBACK_S,
                     release_s: float = DEFAULT_RELEASE_S, **kwargs) -> "AirborneConfig":
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        return cls(lookback_frames=math.ceil(lookback_s * fps), release_frames=math.ceil(release_s * fps), **kwargs)


@dataclass(frozen=True)
class Anchor:
    x_m: float
    y_m: float
    until_frame: int


@dataclass(frozen=True)
class AirborneState:
    history: Mapping[int, tuple[TrackPoint, ...]] = field(default_factory=lambda: MappingProxyType({}))
    anchors: Mapping[int, Anchor] = field(default_factory=lambda: MappingProxyType({}))


def airborne_tracker_ids(player_xyxy: np.ndarray,
                         tracker_ids: np.ndarray | None,
                         action_xyxy: np.ndarray,
                         action_names: Sequence[str],
                         jump_actions: frozenset[str] = DEFAULT_JUMP_ACTIONS,
                         min_iou: float = DEFAULT_MIN_IOU) -> frozenset[int]:
    """Tracker IDs whose box best matches a jump-action box this frame."""
    if len(action_names) != len(action_xyxy):
        raise ValueError(f"{len(action_xyxy)} action boxes but {len(action_names)} names")
    if tracker_ids is None or len(player_xyxy) == 0 or len(action_xyxy) == 0:
        return frozenset()

    players = np.asarray(player_xyxy, dtype=np.float64)
    matched = set()
    for box, name in zip(np.asarray(action_xyxy, dtype=np.float64), action_names):
        if name not in jump_actions:
            continue
        ious = _iou(box, players)
        best = int(np.argmax(ious))
        if ious[best] >= min_iou:
            matched.add(int(tracker_ids[best]))
    return frozenset(matched)


def freeze_airborne(state: AirborneState,
                    positions: Iterable[PlayerPosition],
                    airborne_ids: frozenset[int],
                    frame_index: int,
                    config: AirborneConfig) -> tuple[AirborneState, tuple[PlayerPosition, ...]]:
    """Replace jumping players' positions with their take-off point."""
    window_start = frame_index - config.lookback_frames
    history = {tid: tuple(s for s in samples if s.last_frame >= window_start)
               for tid, samples in state.history.items()}
    anchors = {tid: a for tid, a in state.anchors.items() if a.until_frame >= frame_index}
    out = []

    for p in positions:
        anchor = anchors.get(p.tracker_id)
        if p.tracker_id in airborne_ids:
            anchor = _refresh_anchor(anchor, history.get(p.tracker_id, ()), p, frame_index + config.release_frames)
            anchors[p.tracker_id] = anchor
        if anchor is not None:
            out.append(replace(p, x_m=anchor.x_m, y_m=anchor.y_m, airborne=True))
        else:
            history[p.tracker_id] = (*history.get(p.tracker_id, ()), TrackPoint(p.x_m, p.y_m, frame_index))
            out.append(p)

    new_state = AirborneState(
        history=MappingProxyType({tid: s for tid, s in history.items() if s}),
        anchors=MappingProxyType(anchors),
    )
    return new_state, tuple(out)


def _refresh_anchor(anchor: Anchor | None, grounded: tuple[TrackPoint, ...],
                    current: PlayerPosition, until_frame: int) -> Anchor:
    if anchor is not None:
        return replace(anchor, until_frame=until_frame)
    if grounded:  # earliest grounded sample in the look-back window = before take-off
        take_off = min(grounded, key=lambda s: s.last_frame)
        return Anchor(take_off.x_m, take_off.y_m, until_frame)
    return Anchor(current.x_m, current.y_m, until_frame)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)
