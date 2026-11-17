"""
Cut every non-no-play part of the hand-reviewed range out of a video, then run
ball / player / action detection with a bird's-eye court minimap on each clip.

Usage (from the repo root):
    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m rally_analysis "<video>" [--no-detect]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2

from game_state_clipper.cutter import clip_name, cut_segment, find_ffmpeg
from game_state_clipper.segments import Segment

from .annotations import load_segments, no_play_spans, reviewed_range
from .intervals import keep_spans, states_in

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
logger = logging.getLogger("rally_analysis")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rally_analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--db", type=Path, default=REPO_DIR / "db" / "annotations.db", help="標註資料庫")
    p.add_argument("--calib-db", type=Path, default=REPO_DIR / "db" / "calibration.db", help="場地校正資料庫")
    p.add_argument("--out", type=Path, help="輸出資料夾（預設 output/<影片名>_rallies）")
    p.add_argument("--min-duration", type=float, default=1.0, help="短於此秒數的片段丟棄")
    p.add_argument("--no-detect", action="store_true", help="只剪片，不跑偵測")
    p.add_argument("--only", type=int, nargs="*", help="只跑指定編號的片段（例如 --only 1 2）")
    return p.parse_args(argv)


def fps_of(video: str) -> float:
    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"讀不到影片 FPS：{video}")
    return fps


def plan_segments(args: argparse.Namespace, fps: float) -> list[Segment]:
    manual, model = load_segments(args.db, args.video)
    reviewed = reviewed_range(manual)
    spans = keep_spans(reviewed, no_play_spans(manual), min_len=round(args.min_duration * fps))
    labels = manual + model
    logger.info("已檢查範圍：%.1f s – %.1f s，保留 %d 段", reviewed[0] / fps, reviewed[1] / fps, len(spans))
    return [Segment(s, e, fps, states_in((s, e), labels)) for s, e in spans]


def load_calibration(calib_db: Path, video: str) -> dict:
    from vb_gui.calibration.database import CalibrationDB
    record = CalibrationDB(db_path=calib_db.as_posix()).load(Path(video).as_posix())
    if record is None:
        raise LookupError(f"{calib_db} 裡沒有這部影片的場地校正，請先用校正 GUI 標點")
    return record


def detect_all(clips: list[Path], calibration: dict) -> None:
    os.chdir(SRC_DIR)  # MLManager resolves weight paths relative to src/
    from demo import run_object_detection
    from ml_manager.ml_manager import MLManager
    manager = MLManager()
    for i, clip in enumerate(clips, 1):
        out = clip.with_name(clip.stem + "_detect.mp4")
        logger.info("[%d/%d] 偵測 %s", i, len(clips), clip.name)
        run_object_detection(manager, clip.as_posix(), out.as_posix(), calibration)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.video = str(Path(args.video).resolve())
    fps = fps_of(args.video)
    segments = plan_segments(args, fps)
    out_dir = (args.out or REPO_DIR / "output" / f"{Path(args.video).stem}_rallies").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    clips: list[Path] = []
    for i, seg in enumerate(segments, 1):
        if args.only and i not in args.only:
            continue
        clip = out_dir / clip_name(i, seg)
        if not clip.exists():
            cut_segment(ffmpeg, args.video, seg, clip, copy=False)
        logger.info("  %s (%.1f s)", clip.name, seg.duration_s)
        clips.append(clip)
    (out_dir / "segments.json").write_text(json.dumps(
        [{"index": i, "start_frame": s.start_frame, "end_frame": s.end_frame, "states": list(s.states)}
         for i, s in enumerate(segments, 1)], ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("共 %d 段，總長 %.1f s → %s", len(segments), sum(s.duration_s for s in segments), out_dir)

    if not args.no_detect:
        detect_all(clips, load_calibration(args.calib_db, args.video))
    return 0


if __name__ == "__main__":
    sys.exit(main())
