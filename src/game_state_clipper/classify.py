"""Run the VideoMAE game-state classifier over a whole video with a progress bar."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .segments import Window

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "weights" / "game_state"
MODEL_FRAMES = 16      # VideoMAE input length
MODEL_SIZE = 224       # VideoMAE spatial size


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class ClassifyConfig:
    window_s: float = 1.0          # seconds of video per classification
    start_s: float = 0.0
    end_s: float | None = None     # None = until the end


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"讀不到影片 FPS: {path}")
        return VideoInfo(path, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def sample_offsets(window_len: int, n: int = MODEL_FRAMES) -> frozenset[int]:
    """Uniformly spaced frame offsets inside a window (what VideoMAE will see)."""
    if window_len <= n:
        return frozenset(range(window_len))
    return frozenset(int(round(x)) for x in np.linspace(0, window_len - 1, n))


def pad_frames(frames: list[np.ndarray], n: int = MODEL_FRAMES) -> list[np.ndarray]:
    """VideoMAE needs exactly n frames; repeat the last one for a short tail window."""
    return frames + [frames[-1]] * (n - len(frames)) if 0 < len(frames) < n else list(frames)


def load_classifier(model_dir: Path = DEFAULT_MODEL_DIR):
    # Imported lazily so the CLI's --help and the pure logic don't pay for torch/transformers.
    from ml_manager.models.GameStatusClassifierModule import GameStatusClassifierModule
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"找不到比賽狀態模型: {model_dir}")
    return GameStatusClassifierModule(model_path=model_dir.as_posix(), num_frames=MODEL_FRAMES)


def classify_video(info: VideoInfo, classifier, config: ClassifyConfig,
                   on_progress: Callable[[int, Window], None]) -> list[Window]:
    """Classify consecutive non-overlapping windows; only the sampled frames are decoded."""
    window_len = max(1, round(config.window_s * info.fps))
    first = int(config.start_s * info.fps)
    last = info.total_frames if config.end_s is None else min(info.total_frames, int(config.end_s * info.fps))
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {info.path}")
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    windows: list[Window] = []
    try:
        for start in range(first, last, window_len):
            length = min(window_len, last - start)
            frames, read = _read_window(cap, length, sample_offsets(length))
            if frames:
                result = classifier.classify(pad_frames(frames))
                window = Window(start, start + read, str(result.predicted_class), float(result.confidence))
                windows.append(window)
                on_progress(read, window)
            if read < length:      # stream ended early (frame count in header can be off)
                break
    finally:
        cap.release()
    return windows


def _read_window(cap: cv2.VideoCapture, length: int, offsets: frozenset[int]) -> tuple[list[np.ndarray], int]:
    frames: list[np.ndarray] = []
    for i in range(length):
        if i in offsets:
            ok, frame = cap.read()
            if not ok:
                return frames, i
            frames.append(cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE)))
        elif not cap.grab():
            return frames, i
    return frames, length
