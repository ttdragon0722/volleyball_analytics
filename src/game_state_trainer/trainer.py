"""Fine-tune the VideoMAE game-state classifier and evaluate it the way inference runs."""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import AugmentConfig, ClipDataset, Item, balanced_plan
from .frame_cache import FrameCache
from .labels import CLASSES, Window
from .metrics import Report, evaluate

PREPROCESSOR_FILE = "preprocessor_config.json"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    out_dir: str
    epochs: int = 8
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    samples_per_class: int = 400
    num_workers: int = 2
    seed: int = 42
    grad_checkpointing: bool = False
    reset_head: bool = True        # src/weights/game_state has a degenerate head (|w| up to ~7e3); re-init it
    head_lr: float = 1e-3
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass(frozen=True)
class Split:
    """Caches + candidate windows for a set of videos."""
    videos: tuple[str, ...]
    caches: tuple[FrameCache, ...]
    windows: tuple[tuple[Window, ...], ...]

    def eval_items(self) -> tuple[Item, ...]:
        return tuple(Item(ci, w.start, w.label) for ci, ws in enumerate(self.windows) for w in ws)


def device_and_dtype() -> tuple[torch.device, torch.dtype | None]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), None
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.device("cuda"), amp


def load_model(path: str | Path, device: torch.device):
    from transformers import VideoMAEForVideoClassification

    model_dir = Path(path)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"找不到模型：{model_dir}（需要 HuggingFace 格式的資料夾，裡面要有 config.json）")
    model = VideoMAEForVideoClassification.from_pretrained(model_dir.as_posix(), dtype=torch.float32)
    labels = tuple(model.config.id2label[i].lower() for i in range(model.config.num_labels))
    if labels != CLASSES:
        raise ValueError(f"模型的類別 {labels} 和標註的類別 {CLASSES} 不一致")
    return model.to(device)


@torch.no_grad()
def predict(model, dataset: ClipDataset, device: torch.device, amp: torch.dtype | None,
            batch_size: int = 8, num_workers: int = 2,
            on_batch: Callable[[int, int], None] = lambda d, t: None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds, labels = [], []
    for k, (x, y) in enumerate(loader):
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(pixel_values=x.to(device, non_blocking=True)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(y.numpy())
        on_batch(k + 1, len(loader))
    return np.concatenate(labels), np.concatenate(preds)


def evaluate_split(model, split: Split, num_frames: int, device, amp, num_workers: int = 2,
                   on_batch: Callable[[int, int], None] = lambda d, t: None) -> Report:
    ds = ClipDataset(split.caches, split.eval_items(), num_frames)
    y_true, y_pred = predict(model, ds, device, amp, num_workers=num_workers, on_batch=on_batch)
    return evaluate(y_true, y_pred, CLASSES)


def reset_classifier(model, seed: int) -> None:
    """Fresh classification head (small init) on top of the pretrained backbone."""
    gen = torch.Generator().manual_seed(seed)
    head = model.classifier
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
        head.bias.zero_()


def param_groups(model, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    head = [p for n, p in model.named_parameters() if n.startswith(("classifier.", "fc_norm."))]
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head, "lr": head_lr, "weight_decay": 0.0}]


def save_model(model, base_model: str | Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out.as_posix())
    shutil.copy(Path(base_model) / PREPROCESSOR_FILE, out / PREPROCESSOR_FILE)


@dataclass
class Callbacks:
    on_step: Callable[[int, int, int, float], None] = lambda epoch, step, total, loss: None
    on_eval_batch: Callable[[int, int], None] = lambda done, total: None
    on_epoch: Callable[[int, float, Report, bool], None] = lambda epoch, loss, rep, best: None


def train(cfg: TrainConfig, train_split: Split, val_split: Split, num_frames: int,
          baseline: Report | None = None, callbacks: Callbacks = Callbacks()) -> dict:
    """Fine-tune, keep the best epoch by validation macro-F1, write out_dir/{best,report.json}."""
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    device, amp = device_and_dtype()
    model = load_model(cfg.base_model, device)
    if cfg.reset_head:
        reset_classifier(model, cfg.seed)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    steps_per_epoch = math.ceil(cfg.samples_per_class * len(CLASSES) / cfg.batch_size)
    total_updates = math.ceil(steps_per_epoch * cfg.epochs / cfg.grad_accum)
    optim = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.head_lr if cfg.reset_head else cfg.lr,
                                           cfg.weight_decay))
    sched = get_cosine_schedule_with_warmup(optim, int(total_updates * cfg.warmup_ratio), total_updates)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, best_f1, started = [], -1.0, time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        plan = balanced_plan(train_split.windows, cfg.samples_per_class, cfg.seed + epoch)
        ds = ClipDataset(train_split.caches, plan, num_frames, cfg.augment, seed=cfg.seed + epoch)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda", drop_last=True)
        running, seen = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader, start=1):
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(pixel_values=x.to(device, non_blocking=True)).logits
            loss = loss_fn(logits.float(), y.to(device))
            (loss / cfg.grad_accum).backward()
            if step % cfg.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            running += loss.item() * len(y)
            seen += len(y)
            callbacks.on_step(epoch, step, len(loader), running / seen)
        report = evaluate_split(model, val_split, num_frames, device, amp, cfg.num_workers, callbacks.on_eval_batch)
        is_best = report.macro_f1 > best_f1
        if is_best:
            best_f1 = report.macro_f1
            save_model(model, cfg.base_model, out / "best")
        history.append({"epoch": epoch, "train_loss": round(running / max(1, seen), 4), **report.as_dict()})
        summary = _summary(cfg, num_frames, train_split, val_split, baseline, history, started, epoch == cfg.epochs)
        (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        callbacks.on_epoch(epoch, running / max(1, seen), report, is_best)
    return summary


def _summary(cfg: TrainConfig, num_frames: int, train_split: Split, val_split: Split,
             baseline: Report | None, history: list[dict], started: float, finished: bool) -> dict:
    """report.json content; rewritten after every epoch so an interrupted run still has one."""
    best = max(history, key=lambda h: h["macro_f1"])
    return {
        "finished": finished, "epochs_done": len(history),
        "config": {**asdict(cfg), "num_frames": num_frames},
        "train_videos": list(train_split.videos), "val_videos": list(val_split.videos),
        "baseline": baseline.as_dict() if baseline else None,
        "best_epoch": best["epoch"], "best_macro_f1": best["macro_f1"], "history": history,
        "minutes": round((time.time() - started) / 60, 1),
    }


def build_split(videos: Sequence[str], caches: Sequence[FrameCache],
                windows: Sequence[Sequence[Window]]) -> Split:
    return Split(tuple(videos), tuple(caches), tuple(tuple(w) for w in windows))
