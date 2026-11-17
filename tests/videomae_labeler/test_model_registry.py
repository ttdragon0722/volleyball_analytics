import json
import os
import time

import pytest

from videomae_labeler.model_job import ModelRunner
from videomae_labeler.model_registry import ModelRegistry


def make_model(folder, report=None, mtime=None):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    if mtime is not None:
        os.utime(folder / "config.json", (mtime, mtime))
    if report is not None:
        (folder.parent / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return folder


@pytest.fixture
def layout(tmp_path):
    base = make_model(tmp_path / "weights" / "game_state")
    root = tmp_path / "models"
    now = time.time()
    v1 = make_model(root / "v1" / "best", {"best_macro_f1": 0.79, "val_videos": ["D:/a/2-2.mp4"]}, now - 100)
    v2 = make_model(root / "v2" / "best", {"best_macro_f1": 0.80, "val_videos": ["D:/a/1-3.mp4"]}, now)
    (root / "broken").mkdir()                          # no best/ -> ignored
    return base, root, v1, v2


def test_lists_newest_trained_first_then_original(layout):
    base, root, v1, v2 = layout
    models = ModelRegistry(base, root).list()
    assert [m.label for m in models] == ["v2", "v1", "原始模型（未微調）"]
    assert models[0].macro_f1 == 0.80 and models[0].val_videos == ("1-3.mp4",) and models[0].trained
    assert ModelRegistry(base, root).default_id() == v2.resolve().as_posix()


def test_explicit_model_becomes_default(layout, tmp_path):
    base, root, v1, _ = layout
    extra = make_model(tmp_path / "elsewhere" / "mine")
    reg = ModelRegistry(base, root, extra=extra)
    assert reg.default_id() == extra.resolve().as_posix()
    assert extra.resolve().as_posix() in [m.id for m in reg.list()]
    assert ModelRegistry(base, root, extra=v1).default_id() == v1.resolve().as_posix()
    assert len(ModelRegistry(base, root, extra=v1).list()) == 3          # not listed twice


def test_resolve_only_known_models(layout, tmp_path):
    base, root, v1, v2 = layout
    reg = ModelRegistry(base, root)
    assert reg.resolve(None) == v2.resolve().as_posix()
    assert reg.resolve(str(v1)) == v1.resolve().as_posix()
    with pytest.raises(LookupError):
        reg.resolve(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path / "x", tmp_path / "y").resolve(None)


def test_runner_uses_selected_model(layout):
    base, root, v1, _ = layout
    used = []

    def classify(_path, model_dir, _progress):
        used.append(model_dir)
        return []

    runner = ModelRunner(ModelRegistry(base, root), classify_fn=classify)
    runner.start("v.mp4", str(v1))
    for _ in range(200):
        if runner.status("v.mp4").state != "running":
            break
        time.sleep(0.01)
    assert used == [v1.resolve().as_posix()]
    assert runner.status("v.mp4").model == v1.resolve().as_posix()
    with pytest.raises(LookupError):
        runner.start("v.mp4", "Z:/not/a/model")
