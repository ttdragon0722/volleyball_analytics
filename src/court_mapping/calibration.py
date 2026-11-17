"""
Swappable calibration strategies.

Downstream code only asks "what is the homography for this frame?". Phase 1 uses a
single manual calibration for the whole video; later phases can add providers that
track camera motion or detect court keypoints per frame without touching the pipeline.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from court_mapping.homography import DEFAULT_RANSAC_THRESHOLD_PX, CalibrationError, CourtHomography, fit_homography


class HomographyProvider(Protocol):
    def homography_at(self, frame_index: int, frame: np.ndarray | None = None) -> CourtHomography | None:
        """Return the homography for this frame, or None if it cannot be determined."""


@dataclass(frozen=True)
class StaticHomographyProvider:
    """Fixed camera: one homography for every frame."""
    homography: CourtHomography

    def homography_at(self, frame_index: int, frame: np.ndarray | None = None) -> CourtHomography:
        return self.homography


def provider_from_calibration(calibration: Mapping[str, Any],
                              ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX
                              ) -> StaticHomographyProvider:
    """Build a static provider from a calibration GUI / CalibrationDB record."""
    points = calibration.get("points")
    if not isinstance(points, Mapping):
        raise CalibrationError("Calibration record has no 'points' mapping; re-save it with the calibration GUI.")
    return StaticHomographyProvider(fit_homography(points, ransac_threshold_px))
