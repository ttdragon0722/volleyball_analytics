"""Labelling use-cases: open a video, save the user's timeline."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import timeline as tl
from .store import SegmentStore, media_key

VIDEO_EXTS = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv"})


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def probe_video(path: str) -> VideoInfo:
    import cv2  # imported lazily: heavy, and not needed by tests of the pure logic

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"找不到影片：{path}")
    if file.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"不支援的影片格式：{file.suffix}")
    cap = cv2.VideoCapture(str(file))
    try:
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片：{path}")
        info = VideoInfo(
            path=media_key(file), fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if info.fps <= 0 or info.frame_count <= 0:
        raise ValueError(f"讀不到影片的 fps / 幀數：{path}")
    return info


class LabelingService:
    def __init__(self, store: SegmentStore, prober: Callable[[str], VideoInfo] = probe_video) -> None:
        self.store = store
        self._prober = prober
        self._videos: dict[str, VideoInfo] = {}
        self._lock = threading.Lock()

    def open(self, path: str) -> VideoInfo:
        info = self._prober(path)
        with self._lock:
            self._videos = {**self._videos, info.path: info}
        return info

    def info(self, path: str) -> VideoInfo:
        return self._videos.get(media_key(path)) or self.open(path)

    def segments(self, path: str) -> tl.Timeline:
        return self.store.load(path)

    def save(self, path: str, segments) -> tl.Timeline:
        """Replace everything stored for this video with `segments` (validated)."""
        info = self.info(path)
        timeline = tl.validate_for_video(segments, info.frame_count)
        with self._lock:
            self.store.replace_all(info.path, info.width, info.height, timeline)
        return timeline
