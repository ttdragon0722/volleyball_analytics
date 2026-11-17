"""
Classify a whole video into SERVICE / PLAY / NO-PLAY and cut every non-no-play part into clips.

Usage (from the repo root):
    .venv\\Scripts\\python.exe -m game_state_clipper "D:/video.mp4"
    .venv\\Scripts\\python.exe -m game_state_clipper "D:/video.mp4" --start 240 --end 300 --out output/test
(`src` must be importable: run with PYTHONPATH=src, or from inside src/.)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

from .classify import DEFAULT_MODEL_DIR, ClassifyConfig, classify_video, load_classifier, probe_video
from .cutter import clip_name, cut_segment, find_ffmpeg, write_reports
from .segments import SegmentConfig, Window, build_segments

console = Console()
STATE_COLORS = {"play": "green", "service": "magenta", "no-play": "red"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="game_state_clipper", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="輸入影片路徑")
    p.add_argument("--out", type=Path, default=None, help="輸出資料夾（預設 output/<影片名>_clips）")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="VideoMAE 權重資料夾")
    p.add_argument("--window", type=float, default=1.0, help="每次分類涵蓋的秒數（預設 1.0）")
    p.add_argument("--start", type=float, default=0.0, help="從第幾秒開始（測試用）")
    p.add_argument("--end", type=float, default=None, help="到第幾秒結束（測試用）")
    p.add_argument("--smooth", type=int, default=3, help="多數決平滑視窗數，奇數，1 = 關閉（預設 3）")
    p.add_argument("--merge-gap", type=float, default=1.5, help="間隔小於此秒數的片段合併（預設 1.5）")
    p.add_argument("--min-duration", type=float, default=1.0, help="短於此秒數的片段丟棄（預設 1.0）")
    p.add_argument("--pad", type=float, default=1.0, help="每段前後各多留幾秒（預設 1.0）")
    p.add_argument("--copy", action="store_true", help="ffmpeg 直接複製串流：極快但切點會對齊關鍵幀")
    p.add_argument("--no-cut", action="store_true", help="只分類並輸出 timeline/segments，不剪片")
    args = p.parse_args(argv)
    if args.window <= 0:
        p.error("--window 必須 > 0")
    if args.smooth < 1 or args.smooth % 2 == 0:
        p.error("--smooth 必須是正奇數")
    if args.end is not None and args.end <= args.start:
        p.error("--end 必須大於 --start")
    if not args.video.is_file():
        p.error(f"找不到影片: {args.video}")
    return args


def _progress() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                    TextColumn("[cyan]{task.percentage:>5.1f}%"), TimeElapsedColumn(),
                    TextColumn("剩餘"), TimeRemainingColumn(), console=console)


def run_classification(args: argparse.Namespace) -> tuple[list[Window], float, int]:
    info = probe_video(str(args.video))
    console.print(f"影片: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.total_frames} 幀 "
                  f"({info.total_frames / info.fps / 60:.1f} 分鐘)")
    console.print("載入 VideoMAE 模型...")
    classifier = load_classifier(args.model)

    config = ClassifyConfig(window_s=args.window, start_s=args.start, end_s=args.end)
    last = info.total_frames if args.end is None else min(info.total_frames, int(args.end * info.fps))
    counts: Counter[str] = Counter()
    with _progress() as progress:
        task = progress.add_task("[green]分類中", total=max(0, last - int(args.start * info.fps)))

        def on_progress(n_frames: int, window: Window) -> None:
            counts[window.state] += 1
            color = STATE_COLORS.get(window.state, "white")
            progress.update(task, advance=n_frames, description=(
                f"[green]分類中 [{color}]{window.state:<8}[/] {window.start_frame / info.fps:7.1f}s "
                f"[dim]play {counts['play']} / service {counts['service']} / no-play {counts['no-play']}"))

        windows = classify_video(info, classifier, config, on_progress)
    return windows, info.fps, info.total_frames


def run_cutting(video: Path, segments, out_dir: Path, copy: bool) -> list[str]:
    ffmpeg = find_ffmpeg()
    names = [clip_name(i, s) for i, s in enumerate(segments, start=1)]
    with _progress() as progress:
        task = progress.add_task("[magenta]剪輯中", total=len(segments))
        for name, seg in zip(names, segments):
            progress.update(task, description=f"[magenta]剪輯中 {name}")
            cut_segment(ffmpeg, str(video), seg, out_dir / name, copy)
            progress.advance(task)
    return names


def print_summary(windows: list[Window], segments, fps: float, out_dir: Path) -> None:
    counts = Counter(w.state for w in windows)
    table = Table(title="比賽狀態統計（每個視窗）")
    for col in ("狀態", "視窗數", "佔比"):
        table.add_column(col)
    for state, n in counts.most_common():
        table.add_row(state, str(n), f"{n / len(windows):.1%}")
    console.print(table)
    total = sum(s.duration_s for s in segments)
    console.print(f"剪出 [bold]{len(segments)}[/] 段，共 {total / 60:.1f} 分鐘 → {out_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out or Path("output") / f"{args.video.stem}_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        windows, fps, total_frames = run_classification(args)
        if not windows:
            console.print("[red]沒有讀到任何影格。")
            return 1
        seg_config = SegmentConfig(pad_before_s=args.pad, pad_after_s=args.pad, merge_gap_s=args.merge_gap,
                                   min_duration_s=args.min_duration, smooth_k=args.smooth)
        segments = build_segments(windows, fps, total_frames, seg_config)
        names = [] if args.no_cut or not segments else run_cutting(args.video, segments, out_dir, args.copy)
        write_reports(out_dir, str(args.video), fps, windows, segments,
                      names or [clip_name(i, s) for i, s in enumerate(segments, start=1)])
        print_summary(windows, segments, fps, out_dir)
        if not segments:
            console.print("[yellow]全部被判為 no-play，沒有可剪的片段。可查看 timeline.json 的信心度。")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        console.print(f"[red]錯誤: {e}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]使用者中斷。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
