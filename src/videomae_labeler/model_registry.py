"""Discover the game-state models the labeller can pre-label with.

- the original weights (src/weights/game_state)
- every fine-tuned model produced by game_state_trainer: <models_root>/<name>/best
- an extra folder passed with `--model`

The default is the explicit `--model` if given, otherwise the newest trained model,
otherwise the original weights. Only discovered folders may be used (the browser
picks by id), so the API never loads an arbitrary path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    id: str                       # absolute posix path of the model folder
    label: str
    macro_f1: float | None = None
    val_videos: tuple[str, ...] = ()
    trained: bool = False


def _is_model(folder: Path) -> bool:
    return (folder / "config.json").is_file()


def _trained_info(best: Path) -> ModelInfo:
    report = best.parent / "report.json"
    macro_f1, val = None, ()
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        macro_f1 = data.get("best_macro_f1")
        val = tuple(Path(v).name for v in data.get("val_videos", ()))
    except (OSError, ValueError):
        pass
    return ModelInfo(best.resolve().as_posix(), best.parent.name, macro_f1, val, True)


class ModelRegistry:
    def __init__(self, base_model: str | Path, models_root: str | Path, extra: str | Path | None = None) -> None:
        self.base_model = Path(base_model)
        self.models_root = Path(models_root)
        self.extra = Path(extra) if extra else None

    def list(self) -> tuple[ModelInfo, ...]:
        """Newest trained model first, then the extra model, then the original weights."""
        trained = []
        if self.models_root.is_dir():
            bests = [d / "best" for d in self.models_root.iterdir() if d.is_dir() and _is_model(d / "best")]
            bests.sort(key=lambda p: (p / "config.json").stat().st_mtime, reverse=True)
            trained = [_trained_info(b) for b in bests]
        out = list(trained)
        if self.extra and _is_model(self.extra) and all(m.id != self.extra.resolve().as_posix() for m in out):
            out.append(ModelInfo(self.extra.resolve().as_posix(), self.extra.name, trained=False))
        if _is_model(self.base_model):
            out.append(ModelInfo(self.base_model.resolve().as_posix(), "原始模型（未微調）"))
        return tuple(out)

    def default_id(self) -> str | None:
        models = self.list()
        if self.extra and _is_model(self.extra):
            return self.extra.resolve().as_posix()
        return models[0].id if models else None

    def resolve(self, model_id: str | None) -> str:
        """The folder to load for `model_id` (None = default). Raises for unknown ids."""
        if not model_id:
            default = self.default_id()
            if default is None:
                raise FileNotFoundError(f"找不到任何比賽狀態模型（{self.base_model}、{self.models_root}）")
            return default
        wanted = Path(model_id).resolve().as_posix()
        if any(m.id == wanted for m in self.list()):
            return wanted
        raise LookupError(f"不在模型清單裡：{model_id}")
