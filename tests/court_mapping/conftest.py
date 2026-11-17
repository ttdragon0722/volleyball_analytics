"""Shared fixtures: a synthetic broadcast-like camera with a known homography."""
import cv2
import numpy as np
import pytest

from court_mapping.court_geometry import GROUND_POINTS_M

# Court-plane quad (metres) -> image quad (pixels, 1920x1080) mimicking a
# camera behind the near end line: far end line is short and high in the image.
_COURT_QUAD_M = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])
_IMAGE_QUAD_PX = np.float32([[760, 300], [1160, 300], [1500, 950], [420, 950]])


@pytest.fixture(scope="session")
def true_court_to_image() -> np.ndarray:
    return cv2.getPerspectiveTransform(_COURT_QUAD_M, _IMAGE_QUAD_PX).astype(np.float64)


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


@pytest.fixture(scope="session")
def calibration_points(true_court_to_image) -> dict[str, list[float]]:
    """Calibration-GUI-shaped `points` dict: 10 exact ground points + 4 net points."""
    names = list(GROUND_POINTS_M)
    image_pts = project(true_court_to_image, np.array([GROUND_POINTS_M[n] for n in names]))
    points = {n: [float(x), float(y)] for n, (x, y) in zip(names, image_pts)}
    # Net points are in the air: deliberately wildly inconsistent with the ground plane.
    points.update({
        "net_top_left": [300.0, 100.0],
        "net_top_right": [1700.0, 100.0],
        "net_bottom_left": [300.0, 200.0],
        "net_bottom_right": [1700.0, 200.0],
    })
    return points
