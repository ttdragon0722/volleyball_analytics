"""Project player detections (feet) onto the court plane and filter off-court people."""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from court_mapping.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from court_mapping.homography import CourtHomography

DEFAULT_COURT_MARGIN_M = 1.0


@dataclass(frozen=True)
class PlayerPosition:
    tracker_id: int
    x_m: float
    y_m: float
    observed: bool = True    # False: not detected this frame, shown at last known position
    airborne: bool = False   # True: jumping, position frozen at the take-off point

    @property
    def side(self) -> str:
        """Which half of the court: 'far' (y < net) or 'near'."""
        return "far" if self.y_m < NET_Y_M else "near"


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each (x1, y1, x2, y2) box: the only point touching the floor."""
    boxes = np.asarray(xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes of shape (N, 4), got {boxes.shape}")
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def project_players(xyxy: np.ndarray,
                    tracker_ids: np.ndarray | None,
                    homography: CourtHomography) -> tuple[PlayerPosition, ...]:
    """Map tracked boxes to court metres. Untracked detections are skipped."""
    if tracker_ids is None or len(xyxy) == 0:
        return ()
    if len(tracker_ids) != len(xyxy):
        raise ValueError(f"{len(xyxy)} boxes but {len(tracker_ids)} tracker ids")

    court = homography.to_court(foot_points(xyxy))
    return tuple(
        PlayerPosition(int(tid), float(x), float(y))
        for tid, (x, y) in zip(tracker_ids, court)
        if np.isfinite(x) and np.isfinite(y)
    )


def is_inside_court(x_m: float, y_m: float, margin_m: float = DEFAULT_COURT_MARGIN_M) -> bool:
    return (-margin_m <= x_m <= COURT_WIDTH_M + margin_m) and (-margin_m <= y_m <= COURT_LENGTH_M + margin_m)


def filter_on_court(players: Iterable[PlayerPosition],
                    margin_m: float = DEFAULT_COURT_MARGIN_M) -> tuple[PlayerPosition, ...]:
    """Drop coaches, referees and spectators whose feet project outside court + margin."""
    return tuple(p for p in players if is_inside_court(p.x_m, p.y_m, margin_m))
