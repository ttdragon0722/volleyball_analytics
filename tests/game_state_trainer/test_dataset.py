from collections import Counter

import numpy as np
import pytest
import torch

from game_state_trainer.dataset import (
    AugmentConfig, ClipDataset, Item, augment_clip, balanced_plan, class_counts, to_pixel_values,
)
from game_state_trainer.labels import Window


class FakeCache:
    def clip(self, start, n):
        return np.full((n, 224, 224, 3), start % 256, dtype=np.uint8)


def wins(*labels):
    return [Window(i, lab, 1.0) for i, lab in enumerate(labels)]


def test_balanced_plan_draws_equally_and_is_deterministic():
    per_cache = [wins(0, 0, 0, 0, 1, 1), wins(2)]
    plan = balanced_plan(per_cache, samples_per_class=10, seed=1)
    assert Counter(i.label for i in plan) == {0: 10, 1: 10, 2: 10}
    assert {i.cache for i in plan if i.label == 2} == {1}
    assert plan == balanced_plan(per_cache, 10, seed=1)
    assert plan != balanced_plan(per_cache, 10, seed=2)


def test_balanced_plan_requires_every_class():
    with pytest.raises(ValueError, match="service"):
        balanced_plan([wins(0, 1)], 5, seed=0)


def test_class_counts():
    assert class_counts([wins(0, 1), wins(1, 2)]) == {"no-play": 1, "play": 2, "service": 1}


def test_pixel_values_shape_and_normalisation():
    x = to_pixel_values(np.full((16, 224, 224, 3), 255, dtype=np.uint8))
    assert x.shape == (16, 3, 224, 224) and x.dtype == torch.float32
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor((1 - 0.485) / 0.229))


def test_augment_keeps_shape_and_is_reproducible():
    clip = np.random.default_rng(0).integers(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    a = augment_clip(clip, np.random.default_rng(7), AugmentConfig())
    b = augment_clip(clip, np.random.default_rng(7), AugmentConfig())
    assert a.shape == clip.shape and a.dtype == np.uint8 and np.array_equal(a, b)


def test_clip_dataset_items():
    ds = ClipDataset([FakeCache()], [Item(0, 3, 2)], num_frames=16, augment=AugmentConfig(), seed=0)
    x, y = ds[0]
    assert x.shape == (16, 3, 224, 224) and y == 2 and len(ds) == 1
