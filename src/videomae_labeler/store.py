"""SQLite repository for game-state segments, compatible with vb_annotator.

Reads and writes the same `media` / `game_state_segments` tables as the Qt
annotator (`src/vb_gui/vb_annotator/database`), so anything labelled here shows
up there and in the VideoMAE exporter / rally_analysis, and vice versa.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .timeline import Segment, Timeline, normalize

# Same DDL SQLAlchemy emits for vb_annotator's schema (only used for a brand-new DB).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id INTEGER NOT NULL, path TEXT NOT NULL, media_type VARCHAR(10) NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, court_calibration TEXT,
    created_at DATETIME, PRIMARY KEY (id), UNIQUE (path)
);
CREATE TABLE IF NOT EXISTS game_state_segments (
    id INTEGER NOT NULL, media_id INTEGER NOT NULL, start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL, state VARCHAR(20) NOT NULL, confidence FLOAT NOT NULL,
    source VARCHAR(10) NOT NULL, created_at DATETIME, PRIMARY KEY (id),
    CONSTRAINT uq_game_state_segment UNIQUE (media_id, start_frame, end_frame),
    FOREIGN KEY(media_id) REFERENCES media (id)
);
"""


def media_key(video_path: str | Path) -> str:
    """The `media.path` value vb_annotator uses: absolute, forward slashes."""
    return Path(video_path).resolve().as_posix()


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")


class SegmentStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=10)
        try:
            con.execute("PRAGMA foreign_keys = ON")
            with con:  # commit on success, rollback on exception
                yield con
        finally:
            con.close()

    def list_media(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT m.path, COUNT(g.id), SUM(g.source = 'manual') FROM media m "
                "LEFT JOIN game_state_segments g ON g.media_id = m.id "
                "WHERE m.media_type = 'video' GROUP BY m.id ORDER BY m.path").fetchall()
        return [{"path": p, "segments": n or 0, "manual": int(manual or 0)} for p, n, manual in rows]

    def load(self, video_path: str) -> Timeline:
        with self._connect() as con:
            media_id = self._media_id(con, media_key(video_path))
            if media_id is None:
                return ()
            rows = con.execute(
                "SELECT start_frame, end_frame, state, source, confidence FROM game_state_segments "
                "WHERE media_id = ? ORDER BY start_frame, end_frame", (media_id,)).fetchall()
        return tuple(Segment(s, e, st, src, conf) for s, e, st, src, conf in rows)

    def replace_all(self, video_path: str, width: int, height: int, timeline: Timeline) -> None:
        """Atomically make the DB hold exactly `timeline` for this video."""
        timeline = normalize(timeline)
        key = media_key(video_path)
        with self._connect() as con:
            media_id = self._media_id(con, key)
            if media_id is None:
                cur = con.execute(
                    "INSERT INTO media (path, media_type, width, height, created_at) "
                    "VALUES (?, 'video', ?, ?, ?)", (key, width, height, _now()))
                media_id = cur.lastrowid
            con.execute("DELETE FROM game_state_segments WHERE media_id = ?", (media_id,))
            now = _now()
            con.executemany(
                "INSERT INTO game_state_segments "
                "(media_id, start_frame, end_frame, state, confidence, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(media_id, s.start_frame, s.end_frame, s.state, s.confidence, s.source, now)
                 for s in timeline])

    @staticmethod
    def _media_id(con: sqlite3.Connection, key: str) -> int | None:
        row = con.execute("SELECT id FROM media WHERE path = ?", (key,)).fetchone()
        return row[0] if row else None
