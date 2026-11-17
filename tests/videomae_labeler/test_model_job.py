import time
from types import SimpleNamespace

from videomae_labeler.model_job import ModelRunner, windows_to_segments
from videomae_labeler.model_registry import ModelRegistry
from videomae_labeler.timeline import MODEL, Segment


def W(start, end, state, conf=0.8):
    return SimpleNamespace(start_frame=start, end_frame=end, state=state, confidence=conf)


def wait_done(runner, path):
    for _ in range(200):
        status = runner.status(path)
        if status.state != "running":
            return status
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_windows_merge_and_drop_no_play():
    windows = [W(0, 30, "play", 0.6), W(30, 60, "play", 1.0), W(60, 90, "no-play"),
               W(90, 120, "no-play"), W(120, 150, "service", 0.9), W(150, 180, "service", 0.9)]
    assert windows_to_segments(windows, smooth_k=1) == (
        Segment(0, 59, "play", MODEL, 0.8), Segment(120, 179, "service", MODEL, 0.9))


def test_smoothing_removes_single_window_flicker():
    windows = [W(0, 30, "play"), W(30, 60, "no-play"), W(60, 90, "play")]
    assert windows_to_segments(windows, smooth_k=3) == (Segment(0, 89, "play", MODEL, 0.8),)


def test_unknown_and_empty():
    assert windows_to_segments([]) == ()
    assert windows_to_segments([W(0, 30, "unknown")], smooth_k=1) == ()


def registry_with_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    return ModelRegistry(base, tmp_path / "trained")


def test_runner_reports_errors(tmp_path):
    def boom(_path, _model, _progress):
        raise RuntimeError("cuda out of memory")

    runner = ModelRunner(registry_with_base(tmp_path), classify_fn=boom)
    runner.start("v.mp4")
    status = wait_done(runner, "v.mp4")
    assert status.state == "error" and "cuda out of memory" in status.error


def test_runner_without_weights_is_unavailable(tmp_path):
    runner = ModelRunner(ModelRegistry(tmp_path / "none", tmp_path / "trained"))
    assert runner.available is False
    assert runner.start("v.mp4").state == "error"
