"""
Bird's-eye player mapping (phase 1: fixed camera).

Pipeline: calibration -> homography -> foot-point projection -> off-court filter
-> per-ID temporal smoothing -> minimap / JSON export.

Everything downstream depends only on a `HomographyProvider`, so the calibration
strategy (manual, feature tracking, keypoint model) can be swapped later.
"""
from court_mapping.calibration import HomographyProvider, StaticHomographyProvider, provider_from_calibration
from court_mapping.homography import CalibrationError, CourtHomography, fit_homography
from court_mapping.pipeline import FrameRecord, MapperConfig, MapperState, map_frame
from court_mapping.projection import PlayerPosition

__all__ = [
    "CalibrationError",
    "CourtHomography",
    "FrameRecord",
    "HomographyProvider",
    "MapperConfig",
    "MapperState",
    "PlayerPosition",
    "StaticHomographyProvider",
    "fit_homography",
    "map_frame",
    "provider_from_calibration",
]
