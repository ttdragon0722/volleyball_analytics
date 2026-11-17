import sqlite3

import numpy as np
import pytest

from game_state_trainer.labels import (
    UNLABELLED, Row, candidate_windows, eval_windows, frame_labels, list_labelled_videos, load_rows,
    media_key, stride_for, window_label,
)
from game_state_trainer.metrics import evaluate, format_report

NP, PL, SV = 0, 1, 2


def test_frame_labels_manual_beats_model_and_later_start_wins():
    rows = [Row(0, 9, "play", "model"), Row(5, 7, "no-play", "manual"),
            Row(10, 14, "service", "manual"), Row(14, 19, "play", "manual")]
    lab = frame_labels(rows, 22)
    assert lab[:5].tolist() == [PL] * 5
    assert lab[5:8].tolist() == [NP] * 3
    assert lab[13] == SV and lab[14] == PL          # later-starting manual row wins the shared frame
    assert lab[20:].tolist() == [UNLABELLED, UNLABELLED]


def test_frame_labels_clips_out_of_range_and_ignores_unknown():
    lab = frame_labels([Row(5, 500, "play", "model"), Row(0, 2, "unknown", "model")], 10)
    assert lab.tolist() == [UNLABELLED] * 5 + [PL] * 5


def test_window_label_majority_and_purity():
    lab = np.array([PL, PL, PL, SV, UNLABELLED], dtype=np.int8)
    assert window_label(lab, 0, 5) == (PL, 0.6)
    assert window_label(np.full(4, UNLABELLED, np.int8), 0, 4) == (UNLABELLED, 0.0)


def test_stride_matches_inference_sampling():
    assert stride_for(60, 1.0, 16) == 4     # 60 frames -> 16 samples ~ every 4th frame
    assert stride_for(30, 1.0, 16) == 2
    assert stride_for(10, 1.0, 16) == 1


def test_candidate_windows_respect_purity():
    lab = np.array([NP] * 40 + [SV] * 12 + [PL] * 40, dtype=np.int8)
    wins = candidate_windows(lab, stride=2, num_frames=4, min_purity=1.0)  # windows span 8 frames
    assert {w.label for w in wins} == {NP, SV, PL}
    assert all(w.purity == 1.0 for w in wins)
    sv = [w for w in wins if w.label == SV]
    assert [w.start for w in sv] == [20, 21, 22]   # stored frames 20..22 -> source 40..51


def test_eval_windows_are_non_overlapping_inference_windows():
    lab = np.array([NP] * 60 + [PL] * 60 + [UNLABELLED] * 60, dtype=np.int8)
    wins = eval_windows(lab, stride=4, num_frames=15, window_len=60)
    assert [(w.start, w.label) for w in wins] == [(0, NP), (15, PL)]


def test_metrics():
    rep = evaluate([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], ("no-play", "play", "service"))
    assert rep.confusion == ((1, 1, 0), (0, 2, 0), (1, 0, 1))
    assert rep.accuracy == round(4 / 6, 4)
    assert rep.recall == (0.5, 1.0, 0.5)
    assert rep.precision == (0.5, round(2 / 3, 4), 1.0)
    assert "macro-F1" in format_report(rep, "val")
    with pytest.raises(ValueError):
        evaluate([0], [0, 1], ("a", "b"))


def test_load_rows_from_annotator_db(tmp_path):
    db = tmp_path / "a.db"
    video = tmp_path / "v.mp4"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE media (id INTEGER PRIMARY KEY, path TEXT, media_type TEXT, width INT, height INT);"
                      "CREATE TABLE game_state_segments (id INTEGER PRIMARY KEY, media_id INT, start_frame INT,"
                      " end_frame INT, state TEXT, confidence REAL, source TEXT);")
    con.execute("INSERT INTO media VALUES (1, ?, 'video', 10, 10)", (media_key(video),))
    con.execute("INSERT INTO game_state_segments VALUES (1, 1, 0, 9, 'play', 1.0, 'manual')")
    con.commit()
    con.close()
    assert load_rows(db, video) == (Row(0, 9, "play", "manual"),)
    assert list_labelled_videos(db) == [(media_key(video), 1)]
    with pytest.raises(LookupError):
        load_rows(db, tmp_path / "other.mp4")
    with pytest.raises(FileNotFoundError):
        load_rows(tmp_path / "missing.db", video)
