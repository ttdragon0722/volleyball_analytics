import json

import numpy as np
import pytest
import torch

from game_state_trainer.labels import Window
from game_state_trainer.trainer import (
    TrainConfig, build_split, load_model, param_groups, reset_classifier, train,
)

NUM_FRAMES = 4


class FakeCache:
    """Class-dependent brightness so a tiny model can learn something."""
    def __init__(self, n=60):
        self.n = n

    def clip(self, start, num_frames):
        level = (start // 20) * 100
        return np.full((num_frames, 224, 224, 3), level, dtype=np.uint8)


@pytest.fixture
def tiny_model_dir(tmp_path):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    cfg = VideoMAEConfig(image_size=224, patch_size=56, num_frames=NUM_FRAMES, tubelet_size=2,
                         hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                         num_labels=3, id2label={0: "no-play", 1: "play", 2: "service"},
                         label2id={"no-play": 0, "play": 1, "service": 2}, use_mean_pooling=True)
    model = VideoMAEForVideoClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.fill_(5000.0)          # mimic the degenerate head of the real weights
    out = tmp_path / "base"
    model.save_pretrained(out.as_posix())
    (out / "preprocessor_config.json").write_text(json.dumps({"size": {"shortest_edge": 224}}))
    return out


def windows():
    # stored frames 0-19 -> no-play, 20-39 -> play, 40-59 -> service
    return tuple(Window(s, s // 20, 1.0) for s in range(0, 56, 2))


def test_reset_classifier_and_param_groups(tiny_model_dir):
    model = load_model(tiny_model_dir, torch.device("cpu"))
    reset_classifier(model, seed=0)
    assert model.classifier.weight.abs().max() < 1.0 and torch.all(model.classifier.bias == 0)
    backbone, head = param_groups(model, 1e-5, 1e-3, 0.05)
    assert head["lr"] == 1e-3 and backbone["lr"] == 1e-5
    n_all = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in backbone["params"]) + sum(p.numel() for p in head["params"]) == n_all


def test_load_model_rejects_wrong_classes(tmp_path):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    cfg = VideoMAEConfig(image_size=224, patch_size=56, num_frames=4, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=64, num_labels=2,
                         id2label={0: "a", 1: "b"}, label2id={"a": 0, "b": 1})
    VideoMAEForVideoClassification(cfg).save_pretrained((tmp_path / "m").as_posix())
    with pytest.raises(ValueError, match="類別"):
        load_model(tmp_path / "m", torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing", torch.device("cpu"))


def test_train_end_to_end_writes_hf_model_and_report(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("game_state_trainer.trainer.device_and_dtype", lambda: (torch.device("cpu"), None))
    split = build_split(["v.mp4"], [FakeCache()], [windows()])
    cfg = TrainConfig(base_model=str(tiny_model_dir), out_dir=str(tmp_path / "out"), epochs=2,
                      batch_size=4, grad_accum=1, samples_per_class=4, num_workers=0, lr=1e-3)
    epochs = []
    summary = train(cfg, split, split, NUM_FRAMES,
                    callbacks=__import__("game_state_trainer.trainer", fromlist=["Callbacks"]).Callbacks(
                        on_epoch=lambda e, loss, rep, best: epochs.append((e, loss))))
    assert [e for e, _ in epochs] == [1, 2]
    assert all(loss < 10 for _, loss in epochs)       # reset head: no exploding loss
    best = tmp_path / "out" / "best"
    assert (best / "config.json").is_file() and (best / "preprocessor_config.json").is_file()
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["best_epoch"] in (1, 2) and len(report["history"]) == 2 and report["finished"] is True
    reloaded = load_model(best, torch.device("cpu"))
    assert reloaded.config.id2label[2] == "service"
