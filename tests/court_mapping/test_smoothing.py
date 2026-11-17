import pytest

from court_mapping.projection import PlayerPosition
from court_mapping.smoothing import SmootherState, smooth_positions


def test_first_observation_passes_through():
    state, out = smooth_positions(SmootherState(), (PlayerPosition(1, 2.0, 3.0),), frame_index=0, alpha=0.5)
    assert out == (PlayerPosition(1, 2.0, 3.0),)
    assert 1 in state.tracks


def test_exponential_smoothing_and_immutability():
    initial = SmootherState()
    s1, _ = smooth_positions(initial, (PlayerPosition(1, 0.0, 0.0),), frame_index=0, alpha=0.5)
    s2, out = smooth_positions(s1, (PlayerPosition(1, 4.0, 2.0),), frame_index=1, alpha=0.5)

    assert out[0].x_m == pytest.approx(2.0)
    assert out[0].y_m == pytest.approx(1.0)
    assert initial.tracks == {}
    assert s1.tracks[1].x_m == 0.0  # earlier state untouched
    assert s2 is not s1


def test_stale_tracks_are_reset_and_pruned():
    s1, _ = smooth_positions(SmootherState(), (PlayerPosition(1, 0.0, 0.0),), frame_index=0, alpha=0.5)
    # ID 1 reappears after a long gap -> treated as fresh, not blended with old position.
    s2, out = smooth_positions(s1, (PlayerPosition(1, 8.0, 8.0),), frame_index=100, alpha=0.5, max_gap_frames=10)
    assert out[0].x_m == pytest.approx(8.0)
    # A track that disappears for too long is dropped from the state.
    s3, _ = smooth_positions(s2, (), frame_index=200, alpha=0.5, max_gap_frames=10)
    assert s3.tracks == {}


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        smooth_positions(SmootherState(), (), frame_index=0, alpha=alpha)
