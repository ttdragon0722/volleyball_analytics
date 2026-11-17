"""PyTorch datasets over cached frames: balanced training draws + inference-style eval windows."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .frame_cache import FRAME_SIZE, FrameCache
from .labels import CLASSES, Window

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Item:
    cache: int          # index into the caches list
    start: int          # stored-frame index
    label: int


@dataclass(frozen=True)
class AugmentConfig:
    flip_p: float = 0.5
    brightness: float = 0.2        # +-20 %
    contrast: float = 0.2
    min_crop_area: float = 0.8     # random crop keeps >= 80 % of the frame, then back to 224


def balanced_plan(windows_per_cache: Sequence[Sequence[Window]], samples_per_class: int,
                  seed: int) -> tuple[Item, ...]:
    """Draw `samples_per_class` windows per class (with replacement), shuffled."""
    rng = random.Random(seed)
    by_class: dict[int, list[Item]] = {c: [] for c in range(len(CLASSES))}
    for ci, windows in enumerate(windows_per_cache):
        for w in windows:
            by_class[w.label].append(Item(ci, w.start, w.label))
    missing = [CLASSES[c] for c, items in by_class.items() if not items]
    if missing:
        raise ValueError(f"訓練資料裡沒有這些類別的片段：{missing}")
    plan = [rng.choice(items) for items in by_class.values() for _ in range(samples_per_class)]
    rng.shuffle(plan)
    return tuple(plan)


def class_counts(windows_per_cache: Sequence[Sequence[Window]]) -> dict[str, int]:
    counts = {c: 0 for c in CLASSES}
    for windows in windows_per_cache:
        for w in windows:
            counts[CLASSES[w.label]] += 1
    return counts


def augment_clip(clip: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Same random transform for every frame of a (T, H, W, 3) uint8 clip."""
    out = clip
    if rng.random() < cfg.flip_p:
        out = out[:, :, ::-1]
    area = rng.uniform(cfg.min_crop_area, 1.0)
    side = np.sqrt(area)
    h, w = out.shape[1:3]
    ch, cw = max(8, int(h * side)), max(8, int(w * side))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    if (ch, cw) != (h, w):
        out = np.stack([cv2.resize(np.ascontiguousarray(f[y0:y0 + ch, x0:x0 + cw]), (FRAME_SIZE, FRAME_SIZE))
                        for f in out])
    gain = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
    bias = 255.0 * rng.uniform(-cfg.brightness, cfg.brightness) * 0.5
    mean = out.mean()
    shifted = (out.astype(np.float32) - mean) * gain + mean + bias
    return np.clip(shifted, 0, 255).astype(np.uint8)


def to_pixel_values(clip: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 RGB -> (T, 3, H, W) float32, ImageNet-normalised (VideoMAE input)."""
    x = (clip.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


class ClipDataset(Dataset):
    def __init__(self, caches: Sequence[FrameCache], items: Sequence[Item], num_frames: int,
                 augment: AugmentConfig | None = None, seed: int = 0) -> None:
        self.caches = tuple(caches)
        self.items = tuple(items)
        self.num_frames = num_frames
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        clip = self.caches[item.cache].clip(item.start, self.num_frames)
        if self.augment is not None:
            clip = augment_clip(clip, np.random.default_rng((self.seed, idx)), self.augment)
        return to_pixel_values(clip), item.label


def eval_items(windows: Sequence[Window], cache_index: int = 0) -> tuple[Item, ...]:
    return tuple(Item(cache_index, w.start, w.label) for w in windows)
