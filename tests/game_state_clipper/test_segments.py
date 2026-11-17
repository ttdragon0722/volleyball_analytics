import pytest

from game_state_clipper.segments import (
    SegmentConfig,
    Window,
    build_segments,
    smooth_states,
)

FPS = 10.0


def _windows(states, size=10):
    """One window per state, each `size` frames long, back to back."""
    return [Window(i * size, (i + 1) * size, s, 0.9) for i, s in enumerate(states)]


NO_PAD = SegmentConfig(pad_before_s=0, pad_after_s=0, merge_gap_s=0, min_duration_s=0, smooth_k=1)


class TestSmoothStates:
    def test_k1_is_identity(self):
        assert smooth_states(["a", "b", "a"], 1) == ["a", "b", "a"]

    def test_removes_single_flicker(self):
        s = ["no-play", "no-play", "play", "no-play", "no-play"]
        assert smooth_states(s, 3) == ["no-play"] * 5

    def test_keeps_sustained_run(self):
        s = ["no-play", "play", "play", "play", "no-play"]
        assert smooth_states(s, 3) == s

    def test_does_not_mutate_input(self):
        s = ["no-play", "play", "no-play"]
        smooth_states(s, 3)
        assert s == ["no-play", "play", "no-play"]

    def test_even_k_rejected(self):
        with pytest.raises(ValueError):
            smooth_states(["a"], 2)


class TestBuildSegments:
    def test_empty(self):
        assert build_segments([], FPS, 100, NO_PAD) == []

    def test_all_no_play_gives_nothing(self):
        assert build_segments(_windows(["no-play"] * 5), FPS, 50, NO_PAD) == []

    def test_single_active_run(self):
        segs = build_segments(_windows(["no-play", "service", "play", "no-play"]), FPS, 40, NO_PAD)
        assert len(segs) == 1
        assert (segs[0].start_frame, segs[0].end_frame) == (10, 30)
        assert segs[0].states == ("service", "play")

    def test_unknown_treated_as_inactive(self):
        assert build_segments(_windows(["unknown", "unknown"]), FPS, 20, NO_PAD) == []

    def test_two_runs_stay_separate_without_merge(self):
        segs = build_segments(_windows(["play", "no-play", "play"]), FPS, 30, NO_PAD)
        assert [(s.start_frame, s.end_frame) for s in segs] == [(0, 10), (20, 30)]

    def test_small_gap_is_merged(self):
        cfg = SegmentConfig(pad_before_s=0, pad_after_s=0, merge_gap_s=1.0, min_duration_s=0, smooth_k=1)
        segs = build_segments(_windows(["play", "no-play", "play"]), FPS, 30, cfg)
        assert [(s.start_frame, s.end_frame) for s in segs] == [(0, 30)]

    def test_short_segments_dropped(self):
        cfg = SegmentConfig(pad_before_s=0, pad_after_s=0, merge_gap_s=0, min_duration_s=1.5, smooth_k=1)
        segs = build_segments(_windows(["play", "no-play", "play", "play"]), FPS, 40, cfg)
        assert [(s.start_frame, s.end_frame) for s in segs] == [(20, 40)]

    def test_padding_is_clamped_to_video(self):
        cfg = SegmentConfig(pad_before_s=2, pad_after_s=2, merge_gap_s=0, min_duration_s=0, smooth_k=1)
        segs = build_segments(_windows(["play", "no-play", "no-play"]), FPS, 30, cfg)
        assert (segs[0].start_frame, segs[0].end_frame) == (0, 30)

    def test_padding_overlap_merges(self):
        cfg = SegmentConfig(pad_before_s=1, pad_after_s=1, merge_gap_s=0, min_duration_s=0, smooth_k=1)
        segs = build_segments(_windows(["play", "no-play", "play"]), FPS, 30, cfg)
        assert len(segs) == 1

    def test_segment_times(self):
        segs = build_segments(_windows(["no-play", "play"]), FPS, 20, NO_PAD)
        assert segs[0].start_s == pytest.approx(1.0)
        assert segs[0].duration_s == pytest.approx(1.0)
