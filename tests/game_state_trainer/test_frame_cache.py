import pickle

import cv2
import numpy as np
import pytest

from game_state_trainer.frame_cache import FRAME_SIZE, FrameCache, cache_dir_for, open_or_build


@pytest.fixture
def tiny_video(tmp_path):
    """40 frames, 64x48 @ 20 fps; frame k is filled with gray level 5*k."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for k in range(40):
        writer.write(np.full((48, 64, 3), 5 * k, dtype=np.uint8))
    writer.release()
    return str(path)


def test_build_and_read_every_other_frame(tmp_path, tiny_video):
    progress = []
    cache = open_or_build(tmp_path / "cache", tiny_video, stride=2, on_progress=lambda d, t: progress.append(d))
    assert cache.meta.source_frames == 40 and len(cache) == 20 and cache.meta.fps == pytest.approx(20.0)
    first, tenth = cache.frame(0), cache.frame(10)
    assert first.shape == (FRAME_SIZE, FRAME_SIZE, 3) and first.dtype == np.uint8
    assert abs(int(tenth.mean()) - 5 * 20) <= 3        # stored frame 10 = source frame 20
    assert cache.clip(3, 4).shape == (4, FRAME_SIZE, FRAME_SIZE, 3)
    assert progress[-1] == 40
    with pytest.raises(IndexError):
        cache.frame(20)


def test_cache_is_reused_and_pickles_small(tmp_path, tiny_video):
    root = tmp_path / "cache"
    cache = open_or_build(root, tiny_video, stride=2)
    cache.frame(0)                                       # opens the memmap
    again = open_or_build(root, tiny_video, stride=2)
    assert again.directory == cache.directory == cache_dir_for(root, tiny_video, 2)
    blob = pickle.dumps(cache)
    assert len(blob) < 10_000
    assert pickle.loads(blob).frame(1).shape == (FRAME_SIZE, FRAME_SIZE, 3)


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_or_build(tmp_path / "cache", str(tmp_path / "nope.mp4"), stride=2)


def test_frame_cache_requires_meta(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameCache(tmp_path)
