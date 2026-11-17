"""Decode a video once into 224x224 JPEG frames (every `stride`-th source frame).

Resizing matches inference (game_state_clipper / GameStatusClassifierModule):
the full frame is squashed to 224x224 with cv2.resize (no aspect-preserving crop).
A 20-minute 1080p60 video with stride 4 is ~18k frames, ~200 MB on disk.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

FRAME_SIZE = 224
JPEG_QUALITY = 95
CACHE_VERSION = 1


@dataclass(frozen=True)
class CacheMeta:
    video: str
    fps: float
    source_frames: int      # frames actually decoded from the video
    stride: int
    stored_frames: int
    size: int = FRAME_SIZE
    version: int = CACHE_VERSION


def cache_dir_for(root: Path, video: str, stride: int) -> Path:
    src = Path(video)
    st = src.stat()
    key = f"{src.resolve().as_posix()}|{st.st_size}|{st.st_mtime_ns}|{stride}|{FRAME_SIZE}|{CACHE_VERSION}"
    return Path(root) / f"{src.stem[:40]}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def build_cache(video: str, out_dir: Path, stride: int,
                on_progress: Callable[[int, int], None] = lambda done, total: None) -> CacheMeta:
    """Decode `video` into out_dir/{frames.bin, index.npy, meta.json}. Atomic: written to *.part first."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError(f"讀不到影片 fps：{video}")
    tmp = out_dir.with_name(out_dir.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    i = 0
    try:
        with open(tmp / "frames.bin", "wb") as fh:
            while True:
                if i % stride == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    small = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ok:
                        raise RuntimeError(f"JPEG 編碼失敗（第 {i} 幀）")
                    fh.write(buf.tobytes())
                    offsets.append(offsets[-1] + len(buf))
                elif not cap.grab():
                    break
                i += 1
                if i % 600 == 0:
                    on_progress(i, total)
    finally:
        cap.release()
    meta = CacheMeta(video=Path(video).resolve().as_posix(), fps=fps, source_frames=i,
                     stride=stride, stored_frames=len(offsets) - 1)
    np.save(tmp / "index.npy", np.asarray(offsets, dtype=np.int64))
    (tmp / "meta.json").write_text(json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()
    tmp.rename(out_dir)
    on_progress(i, i)
    return meta


class FrameCache:
    """Read-only access to a built cache. Opens files lazily so it pickles cheaply into DataLoader workers."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        raw = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        self.meta = CacheMeta(**raw)
        self._data = None
        self._index = None

    def __len__(self) -> int:
        return self.meta.stored_frames

    def __getstate__(self):
        return {"directory": self.directory, "meta": self.meta, "_data": None, "_index": None}

    def _open(self) -> None:
        if self._data is None:
            self._index = np.load(self.directory / "index.npy")
            self._data = np.memmap(self.directory / "frames.bin", dtype=np.uint8, mode="r")

    def frame(self, i: int) -> np.ndarray:
        """Stored frame i as RGB uint8 (224, 224, 3)."""
        self._open()
        if not 0 <= i < len(self):
            raise IndexError(f"frame {i} out of range 0..{len(self) - 1}")
        buf = np.asarray(self._data[self._index[i]:self._index[i + 1]])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def clip(self, start: int, num_frames: int) -> np.ndarray:
        """num_frames consecutive stored frames -> (T, H, W, 3) uint8."""
        return np.stack([self.frame(start + k) for k in range(num_frames)])


def open_or_build(root: Path, video: str, stride: int,
                  on_progress: Callable[[int, int], None] = lambda d, t: None) -> FrameCache:
    directory = cache_dir_for(root, video, stride)
    if not (directory / "meta.json").is_file():
        build_cache(video, directory, stride, on_progress)
    return FrameCache(directory)
