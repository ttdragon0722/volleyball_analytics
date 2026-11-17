import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from videomae_labeler.app import create_app
from videomae_labeler.model_job import ModelRunner
from videomae_labeler.model_registry import ModelRegistry
from videomae_labeler.service import LabelingService, VideoInfo
from videomae_labeler.store import SegmentStore, media_key

FRAMES = 1000


def W(start, end, state, conf=0.9):
    return SimpleNamespace(start_frame=start, end_frame=end, state=state, confidence=conf)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "match.mp4"
    path.write_bytes(b"\x00" * 64)
    return media_key(path)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "annotations.db"


@pytest.fixture
def client(db):
    def fake_probe(path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到影片：{path}")
        return VideoInfo(path=media_key(path), fps=30.0, frame_count=FRAMES, width=1920, height=1080)

    def fake_classify(_path, _model_dir, on_progress):
        on_progress(0.5)
        return [W(0, 30, "no-play"), W(30, 60, "service"), W(60, 90, "play"), W(90, 120, "play")]

    service = LabelingService(SegmentStore(db), prober=fake_probe)
    base = db.parent / "base_model"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text("{}")
    registry = ModelRegistry(base, db.parent / "trained")
    return TestClient(create_app(db, service, models=ModelRunner(registry, classify_fn=fake_classify)))


def post(client, url, **body):
    res = client.post(url, json=body)
    return res.status_code, res.json()


def S(start, end, state, source="manual", confidence=1.0):
    return dict(start_frame=start, end_frame=end, state=state, source=source, confidence=confidence)


def db_rows(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT start_frame, end_frame, state, confidence, source "
                           "FROM game_state_segments ORDER BY start_frame").fetchall()
    finally:
        con.close()


def test_open_returns_video_and_empty_segments(client, video):
    code, p = post(client, "/api/open", path=video)
    assert code == 200
    assert p["data"]["video"]["frame_count"] == FRAMES
    assert p["data"]["segments"] == [] and p["data"]["model_available"] is True


def test_save_replaces_everything_in_vb_annotator_format(client, video, db):
    full = [S(0, 99, "no-play", "model", 0.0), S(100, 159, "service"), S(160, 999, "play", "model", 0.8)]
    code, _ = post(client, "/api/save", path=video, segments=full)
    assert code == 200
    assert db_rows(db) == [(0, 99, "no-play", 0.0, "model"), (100, 159, "service", 1.0, "manual"),
                           (160, 999, "play", 0.8, "model")]
    post(client, "/api/save", path=video, segments=[S(0, 999, "no-play")])
    assert db_rows(db) == [(0, 999, "no-play", 1.0, "manual")]
    _, p = post(client, "/api/open", path=video)
    assert p["data"]["segments"] == [S(0, 999, "no-play")]


@pytest.mark.parametrize("segments", [
    [S(0, 10, "play"), S(5, 20, "play")],          # overlap
    [S(0, FRAMES, "play")],                         # past the last frame
])
def test_save_rejects_invalid_timelines(client, video, db, segments):
    code, p = post(client, "/api/save", path=video, segments=segments)
    assert code == 400 and p["success"] is False
    assert db_rows(db) == []


@pytest.mark.parametrize("bad", [
    S(0, 10, "rally"), S(-1, 10, "play"), S(0, 10, "play", "ai"), S(0, 10, "play", confidence=2),
])
def test_save_schema_validation(client, video, bad):
    code, _ = post(client, "/api/save", path=video, segments=[bad])
    assert code == 422


def test_model_job_returns_only_play_and_service(client, video):
    code, p = post(client, "/api/model", path=video)
    assert code == 200 and p["data"]["state"] in ("running", "ready")
    for _ in range(100):
        status = client.get("/api/model", params={"path": video}).json()["data"]
        if status["state"] != "running":
            break
        time.sleep(0.02)
    assert status["state"] == "ready"
    assert [(s["start_frame"], s["end_frame"], s["state"], s["source"]) for s in status["segments"]] == [
        (30, 59, "service", "model"), (60, 119, "play", "model")]


def test_media_listing_and_static_page(client, video):
    post(client, "/api/save", path=video, segments=[S(0, 9, "play")])
    assert client.get("/api/media").json()["data"] == [{"path": video, "segments": 1, "manual": 1}]
    assert "VideoMAE Labeler" in client.get("/").text


def test_errors_use_envelope(client):
    code, p = post(client, "/api/open", path="Z:/missing.txt")
    assert code == 404 and p["success"] is False and "找不到" in p["error"]


def test_video_endpoint_rejects_non_video(client, tmp_path):
    secret = tmp_path / "notes.txt"
    secret.write_text("x")
    assert client.get("/api/video", params={"path": str(secret)}).status_code == 404


class FakeDialog:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def pick_video(self, initial_dir=""):
        self.calls.append(initial_dir)
        if self.error:
            raise self.error
        return self.result


def make_client(db, dialog):
    return TestClient(create_app(db, LabelingService(SegmentStore(db)), dialog=dialog))


def test_pick_returns_chosen_path(db, video):
    dialog = FakeDialog(result=video)
    res = make_client(db, dialog).post("/api/pick", json={"dir": "D:/user/Downloads"}).json()
    assert res["data"] == {"path": video} and dialog.calls == ["D:/user/Downloads"]


def test_pick_cancel_and_busy(db):
    assert make_client(db, FakeDialog(result=None)).post("/api/pick", json={}).json()["data"] == {"path": None}
    from videomae_labeler.file_dialog import DialogBusyError
    res = make_client(db, FakeDialog(error=DialogBusyError("busy"))).post("/api/pick", json={})
    assert res.status_code == 409 and res.json()["success"] is False


def test_file_dialog_initial_dir_falls_back(tmp_path):
    from videomae_labeler.file_dialog import FileDialog
    dlg = FileDialog(tmp_path)
    assert dlg._initial_dir("Z:/does/not/exist") == tmp_path.as_posix()
    assert dlg._initial_dir("") == tmp_path.as_posix()


def test_static_files_are_revalidated(client):
    assert client.get("/js/ops.js").headers["cache-control"] == "no-cache"
    assert client.get("/").headers["cache-control"] == "no-cache"


def test_models_listing_and_run_with_model(client, video, db):
    data = client.get("/api/models").json()["data"]
    base = (db.parent / "base_model").resolve().as_posix()
    assert data["default"] == base and data["models"][0]["id"] == base
    code, p = post(client, "/api/model", path=video, model=base)
    assert code == 200 and p["data"]["model"] == base
    code, p = post(client, "/api/model", path=video, model="Z:/unknown")
    assert code == 404 and p["success"] is False
