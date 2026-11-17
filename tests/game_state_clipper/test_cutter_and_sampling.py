from pathlib import Path

import pytest

from game_state_clipper.__main__ import parse_args
from game_state_clipper.classify import MODEL_FRAMES, pad_frames, sample_offsets
from game_state_clipper.cutter import clip_name, ffmpeg_command
from game_state_clipper.segments import Segment

SEG = Segment(start_frame=600, end_frame=900, fps=60.0, states=("service", "play"))


def test_sample_offsets_uniform_and_bounded():
    offs = sample_offsets(60)
    assert len(offs) == MODEL_FRAMES
    assert min(offs) == 0 and max(offs) == 59


def test_sample_offsets_short_window_uses_all():
    assert sample_offsets(5) == frozenset(range(5))


def test_pad_frames_fills_to_model_length():
    import numpy as np
    frames = [np.zeros((2, 2, 3)), np.ones((2, 2, 3))]
    padded = pad_frames(frames)
    assert len(padded) == MODEL_FRAMES and padded[-1] is frames[-1] and len(frames) == 2


def test_clip_name_contains_time_and_states():
    assert clip_name(3, SEG) == "clip_003_00h00m10s-00h00m15s_service+play.mp4"


def test_ffmpeg_reencode_command():
    cmd = ffmpeg_command("ffmpeg", "in.mp4", SEG, Path("o.mp4"), copy=False)
    assert cmd[cmd.index("-ss") + 1] == "10.000"
    assert cmd[cmd.index("-t") + 1] == "5.000"
    assert "libx264" in cmd and cmd[-1] == "o.mp4"


def test_ffmpeg_copy_command():
    cmd = ffmpeg_command("ffmpeg", "in.mp4", SEG, Path("o.mp4"), copy=True)
    assert cmd[cmd.index("-c") + 1] == "copy"


def test_parse_args_rejects_even_smooth(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(SystemExit):
        parse_args([str(video), "--smooth", "2"])


def test_parse_args_rejects_missing_video(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([str(tmp_path / "nope.mp4")])
