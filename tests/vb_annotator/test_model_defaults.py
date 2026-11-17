import importlib.util
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[2] / "src/vb_gui/vb_annotator/services/model_defaults.py"
_spec = importlib.util.spec_from_file_location("model_defaults", _MODULE)
model_defaults = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_defaults)


class FakeDB:
    def __init__(self, paths=None):
        self.paths = dict(paths or {})

    def get_model_path(self, key):
        return self.paths.get(key)

    def set_model_path(self, key, path):
        self.paths[key] = path


def _file(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")
    return p


def test_fills_unset_keys(tmp_path):
    db = FakeDB()
    ball = _file(tmp_path, "ball.pt")
    applied = model_defaults.apply_default_model_paths(db, {"ball": ball})
    assert db.paths["ball"] == ball.as_posix() and applied == {"ball": ball.as_posix()}


def test_keeps_user_choice_that_exists(tmp_path):
    mine = _file(tmp_path, "mine.pt")
    db = FakeDB({"ball": mine.as_posix()})
    model_defaults.apply_default_model_paths(db, {"ball": _file(tmp_path, "default.pt")})
    assert db.paths["ball"] == mine.as_posix()


def test_replaces_broken_path(tmp_path):
    db = FakeDB({"ball": (tmp_path / "gone.pt").as_posix()})
    default = _file(tmp_path, "default.pt")
    model_defaults.apply_default_model_paths(db, {"ball": default})
    assert db.paths["ball"] == default.as_posix()


def test_skips_missing_default(tmp_path):
    db = FakeDB()
    assert model_defaults.apply_default_model_paths(db, {"ball": tmp_path / "nope.pt"}) == {}
    assert "ball" not in db.paths


def test_repo_defaults_point_at_real_files():
    missing = [k for k, p in model_defaults.DEFAULT_MODEL_PATHS.items() if not p.is_file()]
    assert missing == [], f"default weights missing: {missing}"
