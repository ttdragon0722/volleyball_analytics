"""Read game-state segments from the annotator DB (read-only)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .intervals import LabeledSpan, Span

NO_PLAY = "no-play"


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"找不到標註資料庫：{db_path}")
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def load_segments(db_path: Path, video: str) -> tuple[tuple[LabeledSpan, ...], tuple[LabeledSpan, ...]]:
    """(manual, model) segments of `video`, end frames converted to exclusive."""
    with _connect(db_path) as con:
        row = con.execute("SELECT id FROM media WHERE path = ?", (Path(video).as_posix(),)).fetchone()
        if row is None:
            raise LookupError(f"資料庫裡沒有這部影片的標註：{video}")
        rows = con.execute(
            "SELECT start_frame, end_frame, state, source FROM game_state_segments "
            "WHERE media_id = ? ORDER BY start_frame", (row[0],)).fetchall()
    spans = [(LabeledSpan(s, e + 1, state), src) for s, e, state, src in rows]
    manual = tuple(sp for sp, src in spans if src != "model")
    model = tuple(sp for sp, src in spans if src == "model")
    return manual, model


def reviewed_range(manual: tuple[LabeledSpan, ...]) -> Span:
    """From the first to the last hand-labelled frame."""
    if not manual:
        raise ValueError("這部影片還沒有任何手動標註")
    return min(s.start for s in manual), max(s.end for s in manual)


def no_play_spans(manual: tuple[LabeledSpan, ...]) -> tuple[Span, ...]:
    return tuple((s.start, s.end) for s in manual if s.state == NO_PLAY)
