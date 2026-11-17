"""
One-frame step of the court mapping pipeline:
project -> freeze jumping players -> off-court filter -> smooth -> hold briefly-missing players.
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

from court_mapping.airborne import AirborneConfig, AirborneState, freeze_airborne
from court_mapping.calibration import HomographyProvider
from court_mapping.holding import DEFAULT_SUPPRESS_RADIUS_M, HoldState, hold_missing
from court_mapping.projection import DEFAULT_COURT_MARGIN_M, PlayerPosition, filter_on_court, project_players
from court_mapping.smoothing import DEFAULT_ALPHA, DEFAULT_MAX_GAP_FRAMES, SmootherState, smooth_positions

DEFAULT_HOLD_S = 1.0


@dataclass(frozen=True)
class MapperConfig:
    court_margin_m: float = DEFAULT_COURT_MARGIN_M
    smoothing_alpha: float = DEFAULT_ALPHA
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES
    hold_frames: int = 30
    hold_suppress_radius_m: float = DEFAULT_SUPPRESS_RADIUS_M
    airborne: AirborneConfig = field(default_factory=AirborneConfig)

    @classmethod
    def for_fps(cls, fps: float, hold_s: float = DEFAULT_HOLD_S, **kwargs) -> "MapperConfig":
        """Config with time-based windows (hold, jump look-back/release) converted to frames."""
        airborne = AirborneConfig.from_seconds(fps)
        return cls(hold_frames=math.ceil(hold_s * fps), airborne=airborne, **kwargs)


@dataclass(frozen=True)
class MapperState:
    smoother: SmootherState = field(default_factory=SmootherState)
    airborne: AirborneState = field(default_factory=AirborneState)
    hold: HoldState = field(default_factory=HoldState)
    frame_count: int = 0


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    players: tuple[PlayerPosition, ...]

    @property
    def observed_players(self) -> tuple[PlayerPosition, ...]:
        return tuple(p for p in self.players if p.observed)


def map_frame(state: MapperState,
              frame_index: int,
              xyxy: np.ndarray,
              tracker_ids: np.ndarray | None,
              provider: HomographyProvider,
              config: MapperConfig,
              frame: np.ndarray | None = None,
              airborne_ids: frozenset[int] = frozenset()) -> tuple[MapperState, FrameRecord]:
    """
    Map tracked player boxes of one frame to court positions.

    `airborne_ids`: tracker IDs currently performing a jump action (see
    `court_mapping.airborne.airborne_tracker_ids`). The record contains observed
    players plus briefly-missing ones held at their last position (`observed=False`).
    """
    homography = provider.homography_at(frame_index, frame)
    if homography is None:
        return replace(state, frame_count=state.frame_count + 1), FrameRecord(frame_index, ())

    projected = project_players(xyxy, tracker_ids, homography)
    airborne_state, grounded = freeze_airborne(state.airborne, projected, airborne_ids, frame_index, config.airborne)
    on_court = filter_on_court(grounded, config.court_margin_m)
    smoother, smoothed = smooth_positions(
        state.smoother, on_court, frame_index, config.smoothing_alpha, config.max_gap_frames)
    hold, held = hold_missing(state.hold, smoothed, frame_index, config.hold_frames, config.hold_suppress_radius_m)

    new_state = MapperState(smoother, airborne_state, hold, state.frame_count + 1)
    return new_state, FrameRecord(frame_index, smoothed + held)
