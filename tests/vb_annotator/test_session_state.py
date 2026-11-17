import importlib.util
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[2] / "src/vb_gui/vb_annotator/services/session_state.py"
_spec = importlib.util.spec_from_file_location("session_state", _MODULE)
session_state = importlib.util.module_from_spec(_spec)
sys.modules["session_state"] = session_state   # dataclasses look their module up here
_spec.loader.exec_module(session_state)
SessionState = session_state.SessionState


def test_session_file_sits_next_to_db(tmp_path):
    assert session_state.session_file_for(tmp_path / "annotations.db") == tmp_path / "annotations_session.json"


def test_round_trip(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    f = tmp_path / "s.json"
    session_state.save_session(f, SessionState(str(video), 1234))
    assert session_state.load_session(f) == SessionState(str(video), 1234)


def test_missing_file_is_none(tmp_path):
    assert session_state.load_session(tmp_path / "none.json") is None


def test_corrupt_file_is_none(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("{not json", encoding="utf-8")
    assert session_state.load_session(f) is None


def test_deleted_video_is_none(tmp_path):
    f = tmp_path / "s.json"
    session_state.save_session(f, SessionState(str(tmp_path / "gone.mp4"), 5))
    assert session_state.load_session(f) is None


def test_negative_frame_clamped(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    f = tmp_path / "s.json"
    f.write_text(f'{{"video_path": "{video.as_posix()}", "frame": -3}}', encoding="utf-8")
    assert session_state.load_session(f).frame == 0
