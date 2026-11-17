import shutil
import time
from pathlib import Path

import pytest

from videomae_labeler import proxy as proxy_mod
from videomae_labeler.proxy import ProxyManager, parse_progress, proxy_command

TEST_CLIP = Path(__file__).resolve().parents[2] / "data" / "test_clips" / "13_rally_34526.mp4"


@pytest.mark.parametrize("line, expected", [
    ("out_time_us=5000000\n", 0.5),
    ("out_time_ms=20000000", 1.0),
    ("frame=12", None),
    ("out_time_us=N/A", None),
])
def test_parse_progress(line, expected):
    assert parse_progress(line, 10.0) == expected


def test_command_keeps_frame_timing():
    cmd = proxy_command("ffmpeg", "in.mp4", Path("out.mp4"))
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert "libx264" in cmd and cmd[-1] == "out.mp4"


def test_proxy_path_changes_with_file_content(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"1")
    mgr = ProxyManager(tmp_path / "cache")
    first = mgr.proxy_path(str(src))
    src.write_bytes(b"22")
    assert mgr.proxy_path(str(src)) != first
    assert mgr.status(str(src)).state == "none"


def test_missing_ffmpeg_reports_error(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"1")
    monkeypatch.setattr(proxy_mod.shutil, "which", lambda _: None)
    status = ProxyManager(tmp_path / "cache").start(str(src), 10.0)
    assert status.state == "error" and "ffmpeg" in status.error


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not TEST_CLIP.is_file(), reason="needs ffmpeg + test clip")
def test_real_transcode_keeps_frame_count(tmp_path):
    import cv2

    mgr = ProxyManager(tmp_path / "cache")
    mgr.start(str(TEST_CLIP), 10.0)
    deadline = time.time() + 120
    while mgr.status(str(TEST_CLIP)).state == "running" and time.time() < deadline:
        time.sleep(0.2)
    assert mgr.status(str(TEST_CLIP)).state == "ready", mgr.status(str(TEST_CLIP)).error

    def frames(p):
        cap = cv2.VideoCapture(str(p))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 2)
        finally:
            cap.release()

    assert frames(mgr.proxy_path(str(TEST_CLIP))) == frames(TEST_CLIP)
