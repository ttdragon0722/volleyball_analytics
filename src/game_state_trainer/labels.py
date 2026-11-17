"""Per-frame game-state labels from annotations.db, and the 1-second training windows.

Labels are read straight from the `game_state_segments` table (vb_annotator /
videomae_labeler format, inclusive end frames). Overlapping rows written by older
annotator versions are resolved the same way videomae_labeler does: model rows
first, then manual rows, each later-starting row overwriting earlier ones.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("no-play", "play", "service")          # == id2label of src/weights/game_state
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
UNLABELLED = -1


@dataclass(frozen=True)
class Row:
    start_frame: int
    end_frame: int          # inclusive
    state: str
    source: str


@dataclass(frozen=True)
class Window:
    start: int              # first stored-frame index (see FrameCache stride)
    label: int
    purity: float           # share of source frames carrying `label`


def media_key(video: str | Path) -> str:
    return Path(video).resolve().as_posix()


def load_rows(db_path: str | Path, video: str | Path) -> tuple[Row, ...]:
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError(f"找不到標註資料庫：{db}")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id FROM media WHERE path = ?", (media_key(video),)).fetchone()
        if row is None:
            raise LookupError(f"資料庫裡沒有這部影片的標註：{media_key(video)}")
        rows = con.execute("SELECT start_frame, end_frame, state, source FROM game_state_segments "
                           "WHERE media_id = ?", (row[0],)).fetchall()
    finally:
        con.close()
    return tuple(Row(*r) for r in rows)


def list_labelled_videos(db_path: str | Path) -> list[tuple[str, int]]:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return con.execute("SELECT m.path, COUNT(g.id) FROM media m JOIN game_state_segments g "
                           "ON g.media_id = m.id GROUP BY m.id ORDER BY m.path").fetchall()
    finally:
        con.close()


def frame_labels(rows, n_frames: int) -> np.ndarray:
    """int8 class index per frame, UNLABELLED where nothing covers the frame."""
    labels = np.full(n_frames, UNLABELLED, dtype=np.int8)
    ordered = sorted(rows, key=lambda r: (r.source == "manual", r.start_frame, r.end_frame))
    for r in ordered:
        if r.state not in CLASS_INDEX:
            continue
        start, end = max(0, r.start_frame), min(n_frames - 1, r.end_frame)
        if start <= end:
            labels[start:end + 1] = CLASS_INDEX[r.state]
    return labels


def window_label(labels: np.ndarray, start: int, length: int) -> tuple[int, float]:
    """Majority class over labels[start:start+length] and its share (unlabelled counts against it)."""
    chunk = labels[start:start + length]
    if chunk.size == 0:
        return UNLABELLED, 0.0
    counts = np.bincount(chunk[chunk >= 0], minlength=len(CLASSES))
    if counts.sum() == 0:
        return UNLABELLED, 0.0
    best = int(counts.argmax())
    return best, float(counts[best] / chunk.size)


def candidate_windows(labels: np.ndarray, stride: int, num_frames: int,
                      min_purity: float) -> tuple[Window, ...]:
    """Every training window: `num_frames` consecutive stored frames (stored frame i = source frame i*stride)."""
    span = stride * num_frames
    n_stored = len(labels) // stride
    out = []
    for start in range(0, n_stored - num_frames + 1):
        label, purity = window_label(labels, start * stride, span)
        if label != UNLABELLED and purity >= min_purity:
            out.append(Window(start, label, purity))
    return tuple(out)


def eval_windows(labels: np.ndarray, stride: int, num_frames: int, window_len: int) -> tuple[Window, ...]:
    """Consecutive non-overlapping windows exactly like game_state_clipper's inference."""
    out = []
    for src_start in range(0, len(labels) - window_len + 1, window_len):
        label, purity = window_label(labels, src_start, window_len)
        stored_start = src_start // stride
        if label != UNLABELLED and stored_start + num_frames <= len(labels) // stride:
            out.append(Window(stored_start, label, purity))
    return tuple(out)


def stride_for(fps: float, window_s: float, num_frames: int) -> int:
    """Source-frame step so `num_frames` stored frames span ~`window_s` (inference sampling)."""
    window_len = max(1, round(window_s * fps))
    return max(1, round((window_len - 1) / max(1, num_frames - 1)))
