from court_mapping.holding import HoldState, hold_missing
from court_mapping.projection import PlayerPosition


def _step(state, observed, frame, max_hold=30, radius=0.8):
    return hold_missing(state, tuple(observed), frame, max_hold_frames=max_hold, suppress_radius_m=radius)


def test_missing_player_is_held_at_last_position():
    s, held = _step(HoldState(), [PlayerPosition(1, 2.0, 3.0)], 0)
    assert held == ()
    s, held = _step(s, [], 5)
    assert held == (PlayerPosition(1, 2.0, 3.0, observed=False),)


def test_hold_expires_after_max_frames():
    s, _ = _step(HoldState(), [PlayerPosition(1, 2.0, 3.0)], 0)
    _, held = _step(s, [], 30)
    assert len(held) == 1
    s2, held = _step(s, [], 31)
    assert held == ()
    assert s2.last_seen == {}


def test_reappearing_player_is_not_duplicated():
    s, _ = _step(HoldState(), [PlayerPosition(1, 2.0, 3.0)], 0)
    s, _ = _step(s, [], 3)
    _, held = _step(s, [PlayerPosition(1, 2.5, 3.0)], 4)
    assert held == ()


def test_hold_suppressed_when_another_id_is_nearby():
    """Tracker ID switch: old ID vanishes, a new ID appears on the same spot."""
    s, _ = _step(HoldState(), [PlayerPosition(1, 2.0, 3.0)], 0)
    _, held = _step(s, [PlayerPosition(9, 2.3, 3.2)], 1)
    assert held == ()


def test_state_is_immutable():
    s0 = HoldState()
    s1, _ = _step(s0, [PlayerPosition(1, 2.0, 3.0)], 0)
    _step(s1, [], 1)
    assert s0.last_seen == {}
    assert set(s1.last_seen) == {1}
