"""JSON export of per-frame court coordinates for heatmaps / replays / formation analysis."""
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from court_mapping.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M
from court_mapping.homography import CourtHomography
from court_mapping.pipeline import FrameRecord

SCHEMA_VERSION = 2  # v2: per-player "observed" and "airborne" flags
COORD_DECIMALS = 3


def build_export(video_name: str,
                 fps: float,
                 frames: Iterable[FrameRecord],
                 homography: CourtHomography) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video_name": video_name,
        "fps": fps,
        "court": {"length_m": COURT_LENGTH_M, "width_m": COURT_WIDTH_M, "origin": "far_end_line_left"},
        "calibration": {
            "mean_reprojection_error_px": round(homography.mean_error_px, COORD_DECIMALS),
            "max_reprojection_error_px": round(homography.max_error_px, COORD_DECIMALS),
            "inlier_points": list(homography.inlier_names),
            "image_to_court": homography.image_to_court.tolist(),
        },
        "frames": [_frame_to_dict(f) for f in frames],
    }


def write_export(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return target


def _frame_to_dict(record: FrameRecord) -> dict[str, Any]:
    return {
        "frame": record.frame_index,
        "players": [
            {"id": p.tracker_id, "x": round(p.x_m, COORD_DECIMALS), "y": round(p.y_m, COORD_DECIMALS),
             "side": p.side, "observed": p.observed, "airborne": p.airborne}
            for p in record.players
        ],
    }
