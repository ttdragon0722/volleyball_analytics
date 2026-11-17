"""
Web-based VideoMAE game-state labeller (SERVICE / PLAY / NO-PLAY).

Usage (from the repo root):
    .venv\\Scripts\\python.exe -m videomae_labeler            # then open http://127.0.0.1:8765
    .venv\\Scripts\\python.exe -m videomae_labeler --port 9000 --db D:/other/annotations.db
(`src` must be importable: run with PYTHONPATH=src, or from inside src/.)

Labels go into the same annotations.db as vb_annotator (override with VB_ANNOTATIONS_DB).
"""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "db" / "annotations.db"
DEFAULT_VIDEO_DIR = "D:/user/Downloads"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="videomae_labeler", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.environ.get("VB_ANNOTATIONS_DB", str(DEFAULT_DB)))
    p.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: the API can read local files")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default=None,
                   help="預設選用的模型資料夾（預設：output/game_state_models 裡最新的模型，沒有的話用 src/weights/game_state）")
    p.add_argument("--video-dir", default=os.environ.get("VB_VIDEO_DIR", DEFAULT_VIDEO_DIR),
                   help="folder the open-file dialog starts in (default: %(default)s)")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    import uvicorn

    from .app import DEFAULT_MODEL_DIR, DEFAULT_MODELS_ROOT, create_app
    from .file_dialog import FileDialog
    from .model_job import ModelRunner
    from .model_registry import ModelRegistry

    url = f"http://{args.host}:{args.port}"
    print(f"VideoMAE Labeler → {url}   (DB: {Path(args.db).resolve()})")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()  # after the server is up
    registry = ModelRegistry(DEFAULT_MODEL_DIR, DEFAULT_MODELS_ROOT, extra=args.model)
    app = create_app(args.db, models=ModelRunner(registry),
                     dialog=FileDialog(args.video_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
