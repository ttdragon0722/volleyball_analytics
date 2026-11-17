"""Image <-> court-plane homography fitted from calibration GUI points."""
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np

from court_mapping.court_geometry import GROUND_POINTS_M, MIN_POINTS_FOR_HOMOGRAPHY

DEFAULT_RANSAC_THRESHOLD_PX = 5.0


class CalibrationError(ValueError):
    """Raised when calibration points cannot produce a valid homography."""


@dataclass(frozen=True)
class CourtHomography:
    image_to_court: np.ndarray
    court_to_image: np.ndarray
    reprojection_errors_px: Mapping[str, float]
    inlier_names: tuple[str, ...]

    @property
    def mean_error_px(self) -> float:
        return _mean([self.reprojection_errors_px[n] for n in self.inlier_names])

    @property
    def max_error_px(self) -> float:
        return max((self.reprojection_errors_px[n] for n in self.inlier_names), default=0.0)

    def to_court(self, image_points: np.ndarray) -> np.ndarray:
        """(N, 2) pixel coordinates -> (N, 2) court metres."""
        return _transform(self.image_to_court, image_points)

    def to_image(self, court_points: np.ndarray) -> np.ndarray:
        """(N, 2) court metres -> (N, 2) pixel coordinates."""
        return _transform(self.court_to_image, court_points)


def fit_homography(points: Mapping[str, Sequence[float]],
                   ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX) -> CourtHomography:
    """
    Fit the court homography from the calibration GUI's `points` dict
    ({name: [x_px, y_px]} at original video resolution). Non-ground points are ignored.
    """
    names = tuple(n for n in GROUND_POINTS_M if n in points)
    if len(names) < MIN_POINTS_FOR_HOMOGRAPHY:
        raise CalibrationError(
            f"Need at least {MIN_POINTS_FOR_HOMOGRAPHY} ground points, got {len(names)}: {list(names)}")

    image_pts = np.array([_validate_point(n, points[n]) for n in names], dtype=np.float64)
    court_pts = np.array([GROUND_POINTS_M[n] for n in names], dtype=np.float64)

    # Fit court -> image so the RANSAC threshold is expressed in pixels.
    method = cv2.RANSAC if len(names) > MIN_POINTS_FOR_HOMOGRAPHY else 0
    court_to_image, mask = cv2.findHomography(court_pts, image_pts, method, ransac_threshold_px)
    if court_to_image is None or not _is_invertible(court_to_image):
        raise CalibrationError("Calibration points are degenerate (collinear or duplicated).")

    reprojected = _transform(court_to_image, court_pts)
    errors = np.linalg.norm(reprojected - image_pts, axis=1)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(names), bool)

    return CourtHomography(
        image_to_court=np.linalg.inv(court_to_image),
        court_to_image=court_to_image,
        reprojection_errors_px=MappingProxyType({n: float(e) for n, e in zip(names, errors)}),
        inlier_names=tuple(n for n, ok in zip(names, inliers) if ok),
    )


def _validate_point(name: str, value: Sequence[float]) -> tuple[float, float]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise CalibrationError(f"Point '{name}' must be [x, y], got {value!r}")
    x, y = float(value[0]), float(value[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise CalibrationError(f"Point '{name}' must be finite, got {value!r}")
    return x, y


def _is_invertible(matrix: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(matrix))) and abs(np.linalg.det(matrix)) > 1e-12


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
