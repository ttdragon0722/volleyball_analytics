import numpy as np
import pytest

from court_mapping.homography import fit_homography
from court_mapping.projection import (
    PlayerPosition,
    filter_on_court,
    foot_points,
    is_inside_court,
    project_players,
)
from tests.court_mapping.conftest import project


def test_foot_point_is_bottom_center():
    xyxy = np.array([[100, 50, 140, 250], [0, 0, 10, 20]], dtype=float)
    np.testing.assert_allclose(foot_points(xyxy), [[120, 250], [5, 20]])


def test_foot_points_rejects_wrong_shape():
    with pytest.raises(ValueError):
        foot_points(np.zeros((3, 3)))


def test_project_players_uses_feet(calibration_points, true_court_to_image):
    homography = fit_homography(calibration_points)
    feet_px = project(true_court_to_image, np.array([[2.0, 4.0], [7.0, 14.0]]))
    xyxy = np.array([[fx - 20, fy - 150, fx + 20, fy] for fx, fy in feet_px])

    players = project_players(xyxy, np.array([3, 8]), homography)

    assert [p.tracker_id for p in players] == [3, 8]
    np.testing.assert_allclose([[p.x_m, p.y_m] for p in players], [[2, 4], [7, 14]], atol=1e-3)


def test_project_players_handles_empty_and_missing_ids(calibration_points):
    homography = fit_homography(calibration_points)
    assert project_players(np.zeros((0, 4)), np.zeros(0), homography) == ()
    with pytest.raises(ValueError):
        project_players(np.zeros((2, 4)), np.array([1]), homography)
    # Untracked detections (tracker_id None) cannot be followed over time -> skipped.
    assert project_players(np.zeros((1, 4)), None, homography) == ()


@pytest.mark.parametrize("x, y, inside", [
    (4.5, 9.0, True), (-0.9, 0.0, True), (9.9, 18.9, True),
    (-1.1, 5.0, False), (4.0, 19.5, False), (4.0, -2.0, False),
])
def test_is_inside_court_with_margin(x, y, inside):
    assert is_inside_court(x, y, margin_m=1.0) is inside


def test_filter_on_court_drops_staff():
    players = (PlayerPosition(1, 4.0, 4.0), PlayerPosition(2, 11.5, 9.0), PlayerPosition(3, 4.0, 17.5))
    assert [p.tracker_id for p in filter_on_court(players, margin_m=1.0)] == [1, 3]


def test_team_side_by_net():
    assert PlayerPosition(1, 4.0, 3.0).side == "far"
    assert PlayerPosition(1, 4.0, 12.0).side == "near"
