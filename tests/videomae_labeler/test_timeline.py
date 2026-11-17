import pytest

from videomae_labeler.timeline import MODEL, Segment, normalize, validate_for_video


def seg(s, e, state="play", source="manual", conf=1.0):
    return Segment(s, e, state, source, conf)


def test_normalize_sorts_and_merges_identical_neighbours():
    assert normalize((seg(10, 19), seg(0, 9))) == (seg(0, 19),)


def test_normalize_merges_same_state_manual_wins_confidence_weighted():
    tl = (seg(0, 9, "service", MODEL, 0.6), seg(10, 39, "service"), seg(40, 49, "service", MODEL, 0.8))
    # (0.6*10 + 1.0*30 + 0.8*10) / 50 = 0.88
    assert normalize(tl) == (seg(0, 49, "service", "manual", 0.88),)


def test_normalize_merges_model_only_as_model():
    tl = (seg(0, 119, "play", MODEL, 0.5), seg(120, 239, "play", MODEL, 0.7))
    assert normalize(tl) == (seg(0, 239, "play", MODEL, 0.6),)


def test_normalize_merges_default_no_play_into_manual_no_play():
    tl = (seg(0, 29, "no-play", MODEL, 0.0), seg(30, 39, "no-play"))
    assert normalize(tl) == (seg(0, 39, "no-play", "manual", 0.25),)


def test_normalize_keeps_different_states_and_gaps_apart():
    tl = (seg(0, 9, "play"), seg(10, 19, "service"), seg(21, 30, "service"))
    assert normalize(tl) == tl


def test_normalize_rejects_overlap():
    with pytest.raises(ValueError):
        normalize((seg(0, 10), seg(5, 15)))


def test_validate_for_video_rejects_out_of_range():
    assert validate_for_video((seg(0, 99),), 100) == (seg(0, 99),)
    with pytest.raises(ValueError):
        validate_for_video((seg(0, 100),), 100)


@pytest.mark.parametrize("bad", [
    dict(start_frame=5, end_frame=4, state="play"),
    dict(start_frame=-1, end_frame=4, state="play"),
    dict(start_frame=0, end_frame=4, state="rally"),
    dict(start_frame=0, end_frame=4, state="play", source="ai"),
    dict(start_frame=0, end_frame=4, state="play", confidence=1.5),
])
def test_segment_validation(bad):
    with pytest.raises(ValueError):
        Segment(**bad)
