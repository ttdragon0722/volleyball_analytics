import numpy as np
import pytest

from court_mapping.homography import CalibrationError, fit_homography
from tests.court_mapping.conftest import project


def test_recovers_court_coordinates_from_image(calibration_points, true_court_to_image):
    homography = fit_homography(calibration_points)
    court_pts = np.array([[4.5, 9.0], [1.0, 3.0], [8.0, 16.5]])
    image_pts = project(true_court_to_image, court_pts)

    np.testing.assert_allclose(homography.to_court(image_pts), court_pts, atol=1e-3)
    np.testing.assert_allclose(homography.to_image(court_pts), image_pts, atol=1e-2)


def test_ignores_net_points(calibration_points):
    homography = fit_homography(calibration_points)
    assert not any(name.startswith("net") for name in homography.reprojection_errors_px)
    assert len(homography.reprojection_errors_px) == 10
    assert homography.max_error_px < 1e-3


def test_works_with_only_four_corners(calibration_points):
    corners = {k: calibration_points[k] for k in (
        "far_end_line_left", "far_end_line_right", "near_end_line_left", "near_end_line_right")}
    homography = fit_homography(corners)
    assert homography.max_error_px < 1e-3


def test_ransac_rejects_single_bad_point(calibration_points):
    noisy = {**calibration_points, "far_attack_line_left": [100.0, 100.0]}
    homography = fit_homography(noisy)
    assert "far_attack_line_left" not in homography.inlier_names
    assert homography.reprojection_errors_px["far_attack_line_left"] > 50
    inlier_errors = [homography.reprojection_errors_px[n] for n in homography.inlier_names]
    assert max(inlier_errors) < 1e-2


def test_input_points_are_not_mutated(calibration_points):
    snapshot = {k: list(v) for k, v in calibration_points.items()}
    fit_homography(calibration_points)
    assert calibration_points == snapshot


@pytest.mark.parametrize("bad_points, message", [
    ({}, "at least 4"),
    ({"far_end_line_left": [1, 2], "far_end_line_right": [3, 4], "net_top_left": [5, 6]}, "at least 4"),
    ({"far_end_line_left": [0, 0], "far_end_line_right": [0, 0],
      "near_end_line_left": [0, 0], "near_end_line_right": [0, 0]}, "degenerate"),
    ({"far_end_line_left": [float("nan"), 0], "far_end_line_right": [1, 0],
      "near_end_line_left": [0, 1], "near_end_line_right": [1, 1]}, "finite"),
    ({"far_end_line_left": [1], "far_end_line_right": [1, 0],
      "near_end_line_left": [0, 1], "near_end_line_right": [1, 1]}, "\\[x, y\\]"),
])
def test_rejects_invalid_calibration(bad_points, message):
    with pytest.raises(CalibrationError, match=message):
        fit_homography(bad_points)
