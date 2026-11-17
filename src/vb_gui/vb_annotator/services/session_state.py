"""Remember where the user stopped (video + frame) so the next launch can resume there.

Stored as a tiny JSON file next to the annotations DB. Reading is forgiving: a missing,
corrupt or stale file (video moved/deleted) simply means "nothing to resume".
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionState:
    video_path: str
    frame: int


def session_file_for(db_path: str | Path) -> Path:
    db = Path(db_path)
    return db.with_name(db.stem + "_session.json")


def save_session(path: Path, state: SessionState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)   # atomic: a crash mid-write never leaves a half file
    except OSError as e:
        logger.warning("Could not save annotator session to %s: %s", path, e)


def load_session(path: Path) -> SessionState | None:
    """The last session, or None if there is none or its video no longer exists."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = SessionState(video_path=str(data["video_path"]), frame=max(0, int(data["frame"])))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("Ignoring unreadable annotator session %s: %s", path, e)
        return None
    return state if Path(state.video_path).is_file() else None
