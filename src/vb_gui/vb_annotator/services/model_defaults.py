"""Default model weights for the AI panel, so nothing has to be picked by hand on a fresh DB.

A path the user picked with the "…" button is kept as long as the file still exists;
a default only fills a key that is unset or whose saved file has disappeared.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

SRC_DIR = Path(__file__).resolve().parents[3]          # <repo>/src
POSE_MODEL = SRC_DIR / "yolo11l-pose.pt"

DEFAULT_MODEL_PATHS: Mapping[str, Path] = {
    "ball": SRC_DIR / "weights" / "ball" / "weights" / "best.pt",
    "actions": SRC_DIR / "weights" / "action" / "weights" / "best.pt",
    "players": POSE_MODEL,            # YOLO pose 'person' maps onto the players layer
    "players_pose": POSE_MODEL,
    # HF folder converted with train/hf_to_checkpoint.py
    "game_state": SRC_DIR / "weights" / "game_state_annotator.pt",
}


class ModelPathStore(Protocol):
    def get_model_path(self, key: str) -> str | None: ...
    def set_model_path(self, key: str, path: str) -> None: ...


def apply_default_model_paths(db: ModelPathStore,
                              defaults: Mapping[str, Path] = DEFAULT_MODEL_PATHS) -> dict[str, str]:
    """Fill unset/broken model paths with existing default files. Returns what was set."""
    applied: dict[str, str] = {}
    for key, default in defaults.items():
        current = db.get_model_path(key)
        if current and Path(current).is_file():
            continue
        if default.is_file():
            db.set_model_path(key, default.as_posix())
            applied[key] = default.as_posix()
    return applied
