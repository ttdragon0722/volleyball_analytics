"""
Convert a HuggingFace VideoMAE folder (config.json + model.safetensors) into the single
.pt checkpoint the annotator's Game State "…" button expects (same format vmae_train.py saves).

Usage (repo root):
    .venv\\Scripts\\python.exe src/vb_gui/vb_annotator/train/hf_to_checkpoint.py
    .venv\\Scripts\\python.exe src/vb_gui/vb_annotator/train/hf_to_checkpoint.py <hf_dir> <out.pt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import VideoMAEForVideoClassification

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SRC = REPO_ROOT / "src" / "weights" / "game_state"
DEFAULT_OUT = REPO_ROOT / "src" / "weights" / "game_state_annotator.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_checkpoint(hf_dir: Path) -> dict:
    if not (hf_dir / "config.json").is_file():
        raise FileNotFoundError(f"{hf_dir} 不是 HuggingFace 模型資料夾（缺 config.json）")
    model = VideoMAEForVideoClassification.from_pretrained(hf_dir.as_posix())
    cfg = model.config
    classes = [cfg.id2label[i] for i in range(cfg.num_labels)]   # keep the head's output order

    pre_path = hf_dir / "preprocessor_config.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.is_file() else {}

    return {
        "version": 1,
        # A local folder works with AutoConfig.from_pretrained, so loading needs no network.
        "model_name": hf_dir.resolve().as_posix(),
        "classes": classes,
        "num_frames": cfg.num_frames,
        "image_size": cfg.image_size,
        "mean": pre.get("image_mean", IMAGENET_MEAN),
        "std": pre.get("image_std", IMAGENET_STD),
        # The annotator wraps the HF model as `self.model`, hence the "model." prefix.
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hf_dir", nargs="?", type=Path, default=DEFAULT_SRC)
    p.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        ckpt = build_checkpoint(args.hf_dir)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"錯誤: {e}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"classes={ckpt['classes']}  num_frames={ckpt['num_frames']}  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
