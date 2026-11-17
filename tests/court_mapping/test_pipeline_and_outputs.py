import json

import numpy as np
import pytest

from court_mapping.calibration import StaticHomographyProvider, provider_from_calibration
from court_mapping.export import build_export, write_export
from court_mapping.homography import CalibrationError, fit_homography
from court_mapping.minimap import MinimapStyle, overlay_minimap, render_minimap
from court_mapping.pipeline import MapperConfig, MapperState, map_frame
from court_mapping.projection import PlayerPosition
from tests.court_mapping.conftest import project


def _boxes_at(true_court_to_image, court_pts):
    feet = project(true_court_to_image, np.array(court_pts, dtype=float))
    return np.array([[fx - 20, fy - 150, fx + 20, fy] for fx, fy in feet])


def test_provider_from_calibration(calibration_points):
    provider = provider_from_calibration({"points": calibration_points})
    assert isinstance(provider, StaticHomographyProvider)
    assert provider.homography_at(0) is provider.homography_at(500)
    with pytest.raises(CalibrationError, match="points"):
        provider_from_calibration({"court_corners": []})


def test_map_frame_filters_and_smooths(calibration_points, true_court_to_image):
    provider = provider_from_calibration({"points": calibration_points})
    xyxy = _boxes_at(true_court_to_image, [[2, 4], [7, 14], [12, 9]])  # 3rd = coach beside court
    config = MapperConfig()

    state, record = map_frame(MapperState(), 0, xyxy, np.array([1, 2, 3]), provider, config)

    assert record.frame_index == 0
    assert [p.tracker_id for p in record.players] == [1, 2]
    assert record.players[0].side == "far" and record.players[1].side == "near"
    assert state.frame_count == 1


def test_map_frame_holds_missing_player_semi_observed(calibration_points, true_court_to_image):
    provider = provider_from_calibration({"points": calibration_points})
    config = MapperConfig(hold_frames=30)
    xyxy = _boxes_at(true_court_to_image, [[2, 4], [7, 14]])

    state, _ = map_frame(MapperState(), 0, xyxy, np.array([1, 2]), provider, config)
    state, record = map_frame(state, 1, xyxy[:1], np.array([1]), provider, config)

    by_id = {p.tracker_id: p for p in record.players}
    assert by_id[1].observed and not by_id[2].observed
    assert by_id[2].x_m == pytest.approx(7, abs=1e-3)


def test_map_frame_freezes_jumping_player(calibration_points, true_court_to_image):
    provider = provider_from_calibration({"points": calibration_points})
    config = MapperConfig(smoothing_alpha=1.0)
    state = MapperState()
    for frame in range(10):  # grounded at y=10
        state, _ = map_frame(state, frame, _boxes_at(true_court_to_image, [[4, 10]]), np.array([5]), provider, config)
    # Blocker in the air: feet project ~4 m deeper into the other half
    state, record = map_frame(state, 10, _boxes_at(true_court_to_image, [[4, 6]]), np.array([5]), provider, config,
                              airborne_ids=frozenset({5}))
    assert record.players[0].y_m == pytest.approx(10, abs=1e-3)
    assert record.players[0].airborne and record.players[0].side == "near"


def test_mapper_config_for_fps():
    config = MapperConfig.for_fps(60)
    assert config.hold_frames == 60
    assert config.airborne.release_frames == 30


def test_map_frame_without_homography_returns_empty(calibration_points):
    class NoHomography:
        def homography_at(self, frame_index, frame=None):
            return None

    _, record = map_frame(MapperState(), 5, np.zeros((1, 4)), np.array([1]), NoHomography(), MapperConfig())
    assert record.players == ()


def test_minimap_render_and_overlay_do_not_mutate():
    style = MinimapStyle()
    minimap = render_minimap((PlayerPosition(7, 4.5, 3.0), PlayerPosition(9, 2.0, 15.0),
                              PlayerPosition(3, 6.0, 15.0, observed=False),
                              PlayerPosition(4, 6.0, 12.0, airborne=True)), style)
    h, w = minimap.shape[:2]
    assert h > w  # court is portrait: 18 m long, 9 m wide
    assert h == int(round((18 + 2 * style.padding_m) * style.pixels_per_metre))

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = overlay_minimap(frame, minimap)
    assert frame.sum() == 0
    assert out.shape == frame.shape and out.sum() > 0


def test_held_player_drawn_dimmer_than_observed():
    style = MinimapStyle()
    from court_mapping.minimap import court_to_minimap_px
    observed = render_minimap((PlayerPosition(1, 4.5, 14.0),), style)
    held = render_minimap((PlayerPosition(1, 4.5, 14.0, observed=False),), style)
    cx, cy = court_to_minimap_px(4.5, 14.0, style)
    floor = np.array(style.floor_color, float)
    dist_observed = np.abs(observed[cy, cx].astype(float) - floor).sum()
    dist_held = np.abs(held[cy, cx].astype(float) - floor).sum()
    assert 0 < dist_held < dist_observed


def test_overlay_rejects_minimap_larger_than_frame():
    with pytest.raises(ValueError):
        overlay_minimap(np.zeros((50, 50, 3), np.uint8), render_minimap((), MinimapStyle()))


def test_export_round_trip(tmp_path, calibration_points):
    homography = fit_homography(calibration_points)
    from court_mapping.pipeline import FrameRecord
    frames = (FrameRecord(0, (PlayerPosition(1, 1.23456, 2.0),)),
              FrameRecord(1, (PlayerPosition(1, 1.0, 2.0, observed=False, airborne=True),)),
              FrameRecord(2, ()))

    data = build_export("rally.mp4", 29.97, frames, homography)
    path = write_export(tmp_path / "out" / "rally_court.json", data)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["video_name"] == "rally.mp4"
    assert loaded["fps"] == 29.97
    assert loaded["court"] == {"length_m": 18.0, "width_m": 9.0, "origin": "far_end_line_left"}
    assert loaded["calibration"]["max_reprojection_error_px"] < 1e-3
    assert loaded["frames"][0]["players"] == [
        {"id": 1, "x": 1.235, "y": 2.0, "side": "far", "observed": True, "airborne": False}]
    assert loaded["frames"][1]["players"][0]["observed"] is False
    assert loaded["frames"][1]["players"][0]["airborne"] is True
    assert loaded["frames"][2]["players"] == []
