"""
Fine-tune the VideoMAE game-state model (service / play / no-play) on your own labelled videos.

Usage (from the repo root):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m game_state_trainer --list
    .venv\\Scripts\\python.exe -m game_state_trainer --train "D:/v/3-1.mp4" --val "D:/v/2-2.mp4"
    .venv\\Scripts\\python.exe -m game_state_trainer --evaluate src/weights/game_state --val "D:/v/2-2.mp4"

Labels are read from annotations.db (videomae_labeler / vb_annotator). Train and validation
must be DIFFERENT videos. The result is a HuggingFace model folder usable with
`videomae_labeler --model <dir>` and `game_state_clipper --model <dir>`.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .frame_cache import open_or_build
from .labels import (
    CLASSES, candidate_windows, eval_windows, frame_labels, list_labelled_videos, load_rows, media_key, stride_for,
)
from .metrics import format_report

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "db" / "annotations.db"
DEFAULT_BASE = REPO / "src" / "weights" / "game_state"
DEFAULT_CACHE = REPO / "output" / "game_state_cache"
DEFAULT_OUT = REPO / "output" / "game_state_models"
NUM_FRAMES = 16
WINDOW_S = 1.0

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="game_state_trainer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="列出資料庫中有標註的影片後結束")
    p.add_argument("--train", nargs="+", default=[], metavar="VIDEO", help="訓練用影片（可多部）")
    p.add_argument("--val", nargs="+", default=[], metavar="VIDEO", help="驗證用影片（必須和訓練影片不同）")
    p.add_argument("--evaluate", metavar="MODEL_DIR", help="只評估這個模型在 --val 影片上的表現，不訓練")
    p.add_argument("--db", default=os.environ.get("VB_ANNOTATIONS_DB", str(DEFAULT_DB)))
    p.add_argument("--base-model", default=str(DEFAULT_BASE), help="從這個模型開始微調（預設：已學過排球的權重）")
    p.add_argument("--out", default=None, help=f"輸出資料夾（預設：{DEFAULT_OUT}/<時間>）")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="梯度累積步數（等效 batch = batch-size × 這個值）")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--samples-per-class", type=int, default=400, help="每個 epoch 每個類別抽幾個 1 秒窗口")
    p.add_argument("--min-purity", type=float, default=0.75, help="訓練窗口裡多數類別至少要佔多少比例")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-checkpointing", action="store_true", help="省顯示卡記憶體（較慢）；遇到 CUDA out of memory 時使用")
    p.add_argument("--skip-baseline", action="store_true", help="不先評估原本的模型")
    p.add_argument("--keep-head", action="store_true",
                   help="沿用原模型的分類層（預設會重新初始化：src/weights/game_state 的分類層權重異常）")
    args = p.parse_args(argv)
    if not args.list:
        if not args.val:
            p.error("需要 --val（驗證用影片）")
        if not args.evaluate and not args.train:
            p.error("需要 --train（訓練用影片），或用 --evaluate 只做評估")
        overlap = {media_key(v) for v in args.train} & {media_key(v) for v in args.val}
        if overlap:
            p.error(f"訓練和驗證不能用同一部影片：{sorted(overlap)}")
        if not 0 < args.min_purity <= 1:
            p.error("--min-purity 必須在 (0, 1] 之間")
    return args


def _progress() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                    TimeElapsedColumn(), TimeRemainingColumn(), console=console)


def prepare(videos, db, cache_root: Path, training: bool, min_purity: float):
    """Frame caches + windows for each video. Training videos get dense candidate windows."""
    caches, windows = [], []
    for video in videos:
        rows = load_rows(db, video)
        cache = _cache(video, cache_root)
        labels = frame_labels(rows, cache.meta.source_frames)
        stride = cache.meta.stride
        if training:
            ws = candidate_windows(labels, stride, NUM_FRAMES, min_purity)
        else:
            ws = eval_windows(labels, stride, NUM_FRAMES, max(1, round(WINDOW_S * cache.meta.fps)))
        counts = {c: sum(1 for w in ws if w.label == i) for i, c in enumerate(CLASSES)}
        unl = int((labels < 0).sum())
        console.print(f"  {Path(video).name}: {cache.meta.source_frames} 幀 @ {cache.meta.fps:.0f} fps，"
                      f"{'候選' if training else '驗證'}窗口 {counts}" + (f"，未標註 {unl} 幀" if unl else ""))
        caches.append(cache)
        windows.append(ws)
    return caches, windows


def _cache(video: str, root: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = stride_for(fps, WINDOW_S, NUM_FRAMES)
    with _progress() as progress:
        task = progress.add_task(f"[cyan]解碼影格（只需一次）{Path(video).name[:30]}", total=None)
        return open_or_build(root, video, stride,
                             on_progress=lambda d, t: progress.update(task, completed=d, total=t or None))


def run_evaluate(args) -> None:
    from .trainer import build_split, device_and_dtype, evaluate_split, load_model

    console.print("[bold]準備驗證資料")
    caches, windows = prepare(args.val, args.db, Path(args.cache_dir), False, args.min_purity)
    device, amp = device_and_dtype()
    model = load_model(args.evaluate, device)
    split = build_split(args.val, caches, windows)
    with _progress() as progress:
        task = progress.add_task("[green]評估", total=None)
        rep = evaluate_split(model, split, NUM_FRAMES, device, amp, args.workers,
                             lambda d, t: progress.update(task, completed=d, total=t))
    console.print(format_report(rep, f"模型 {args.evaluate}"))


def run_train(args) -> None:
    from .trainer import Callbacks, TrainConfig, build_split, device_and_dtype, evaluate_split, load_model, train

    out = Path(args.out) if args.out else DEFAULT_OUT / datetime.now().strftime("%Y%m%d-%H%M")
    console.print("[bold]準備訓練資料")
    tr_caches, tr_windows = prepare(args.train, args.db, Path(args.cache_dir), True, args.min_purity)
    console.print("[bold]準備驗證資料")
    va_caches, va_windows = prepare(args.val, args.db, Path(args.cache_dir), False, args.min_purity)
    train_split = build_split(args.train, tr_caches, tr_windows)
    val_split = build_split(args.val, va_caches, va_windows)
    device, amp = device_and_dtype()
    console.print(f"裝置：{device}（{'混合精度 ' + str(amp).replace('torch.', '') if amp else 'fp32'}）")

    baseline = None
    if not args.skip_baseline:
        with _progress() as progress:
            task = progress.add_task("[yellow]評估原本的模型（基準）", total=None)
            base_model = load_model(args.base_model, device)
            baseline = evaluate_split(base_model, val_split, NUM_FRAMES, device, amp, args.workers,
                                      lambda d, t: progress.update(task, completed=d, total=t))
            del base_model
        console.print(format_report(baseline, "基準（訓練前）"))

    cfg = TrainConfig(base_model=args.base_model, out_dir=str(out), epochs=args.epochs,
                      batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
                      samples_per_class=args.samples_per_class, num_workers=args.workers,
                      seed=args.seed, grad_checkpointing=args.grad_checkpointing, reset_head=not args.keep_head)
    with _progress() as progress:
        task = progress.add_task("訓練", total=None)

        def on_step(epoch, step, total, loss):
            progress.update(task, completed=step, total=total,
                            description=f"[green]epoch {epoch}/{cfg.epochs} loss {loss:.3f}")

        def on_eval(done, total):
            progress.update(task, completed=done, total=total, description="[yellow]驗證中")

        def on_epoch(epoch, loss, rep, best):
            mark = "  [bold green]★ 目前最好，已儲存" if best else ""
            progress.console.print(f"epoch {epoch}: train loss {loss:.3f} | val acc {rep.accuracy:.3f} "
                                   f"macro-F1 {rep.macro_f1:.3f} | service recall {rep.recall[2]:.3f}{mark}")

        summary = train(cfg, train_split, val_split, NUM_FRAMES, baseline, Callbacks(on_step, on_eval, on_epoch))

    best = summary["history"][summary["best_epoch"] - 1]
    console.rule("[bold]完成")
    if baseline:
        console.print(f"基準 macro-F1 {baseline.macro_f1:.3f} → 微調後 {best['macro_f1']:.3f}"
                      f"（第 {summary['best_epoch']} 個 epoch，共 {summary['minutes']} 分鐘）")
    console.print(f"最好的模型：[bold]{out / 'best'}[/]")
    console.print(f"報告：{out / 'report.json'}")
    console.print(f"用新模型預標：.venv\\Scripts\\python.exe -m videomae_labeler --model \"{out / 'best'}\"")


def _quiet_transformers() -> None:
    """Hide HuggingFace's per-load / per-save tqdm bars; our own progress bars stay."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _quiet_transformers()
    try:
        if args.list:
            for path, n in list_labelled_videos(args.db):
                console.print(f"  {n:5d} 段  {path}")
        elif args.evaluate:
            run_evaluate(args)
        else:
            run_train(args)
    except (FileNotFoundError, LookupError, ValueError) as exc:
        console.print(f"[red]錯誤：{exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]已中斷。到目前為止最好的模型（best\）和 report.json 都還在輸出資料夾裡。")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
