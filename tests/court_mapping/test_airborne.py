import numpy as np
import pytest

from court_mapping.airborne import AirborneConfig, AirborneState, airborne_tracker_ids, freeze_airborne
from court_mapping.projection import PlayerPosition

CFG = AirborneConfig(lookback_frames=3, release_frames=4)


def test_airborne_ids_match_jump_actions_to_players():
    players = np.array([[100, 100, 200, 400], [500, 100, 600, 400], [900, 100, 1000, 400]], float)
    ids = np.array([7, 8, 9])
    actions = np.array([[105, 90, 205, 390], [495, 110, 600, 410]], float)
    names = ["block", "receive"]
    assert airborne_tracker_ids(players, ids, actions, names) == frozenset({7})


def test_airborne_ids_ignores_weak_overlap_and_handles_empty():
    players = np.array([[100, 100, 200, 400]], float)
    far_action = np.array([[600, 100, 700, 400]], float)
    assert airborne_tracker_ids(players, np.array([1]), far_action, ["spike"]) == frozenset()
    assert airborne_tracker_ids(players, None, far_action, ["spike"]) == frozenset()
    assert airborne_tracker_ids(np.zeros((0, 4)), np.zeros(0), far_action, ["spike"]) == frozenset()
    assert airborne_tracker_ids(players, np.array([1]), np.zeros((0, 4)), []) == frozenset()


def test_airborne_ids_rejects_mismatched_names():
    with pytest.raises(ValueError):
        airborne_tracker_ids(np.zeros((1, 4)), np.array([1]), np.zeros((2, 4)), ["block"])


def _run(sequence):
    """sequence: list of (y_m, airborne?) for tracker 1; returns output y per frame."""
    state, out = AirborneState(), []
    for frame, (y, airborne) in enumerate(sequence):
        ids = frozenset({1}) if airborne else frozenset()
        state, positions = freeze_airborne(state, (PlayerPosition(1, 4.0, y),), ids, frame, CFG)
        out.append((positions[0].y_m, positions[0].airborne))
    return out


def test_position_frozen_at_pre_jump_point_while_airborne():
    # Grounded at 10.0, jump starts at frame 3 (projection drifts away), action detected at frame 5.
    seq = [(10.0, False), (10.0, False), (10.0, False), (9.0, False), (8.0, False),
           (7.0, True), (6.0, True), (6.5, False), (7.5, False)]
    out = _run(seq)
    # lookback 3 frames from frame 5 -> grounded sample from frame 2 (10.0), not the drifting 9.0/8.0
    assert out[5] == (10.0, True)
    assert out[6] == (10.0, True)
    # release window (4 frames) keeps it frozen during landing
    assert out[7] == (10.0, True) and out[8] == (10.0, True)


def test_position_released_after_window():
    seq = [(10.0, False), (10.0, True)] + [(12.0, False)] * 6
    out = _run(seq)
    assert out[5] == (10.0, True)       # frame 1 + 4 release frames
    assert out[6] == (12.0, False)


def test_first_seen_while_airborne_uses_current_position():
    out = _run([(5.0, True), (5.0, False)])
    assert out[0] == (5.0, True)


def test_other_players_pass_through_and_state_immutable():
    s0 = AirborneState()
    s1, pos = freeze_airborne(s0, (PlayerPosition(2, 1.0, 1.0),), frozenset({99}), 0, CFG)
    assert pos == (PlayerPosition(2, 1.0, 1.0),)
    assert s0.history == {} and 2 in s1.history


def test_config_from_seconds():
    cfg = AirborneConfig.from_seconds(fps=30, lookback_s=0.25, release_s=0.5)
    assert (cfg.lookback_frames, cfg.release_frames) == (8, 15)
    with pytest.raises(ValueError):
        AirborneConfig.from_seconds(fps=0)
