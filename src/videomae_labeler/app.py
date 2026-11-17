"""FastAPI app: JSON API + static web UI for the VideoMAE game-state labeller."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import timeline as tl
from .file_dialog import DialogBusyError, FileDialog
from .model_job import ModelRunner
from .model_registry import ModelRegistry
from .proxy import ProxyManager
from .service import VIDEO_EXTS, LabelingService
from .store import SegmentStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "weights" / "game_state"
DEFAULT_MODELS_ROOT = Path(__file__).resolve().parents[2] / "output" / "game_state_models"
MAX_SEGMENTS = 200_000


class PathBody(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ModelRunBody(PathBody):
    model: str | None = Field(default=None, max_length=4096)


class PickBody(BaseModel):
    dir: str = Field(default="", max_length=4096)


class SegmentBody(BaseModel):
    start_frame: int = Field(ge=0, le=2**31 - 1)
    end_frame: int = Field(ge=0, le=2**31 - 1)
    state: Literal["service", "play", "no-play"]
    source: Literal["manual", "model"]
    confidence: float = Field(ge=0.0, le=1.0)


class SaveBody(PathBody):
    segments: list[SegmentBody] = Field(max_length=MAX_SEGMENTS)


def _ok(data) -> dict:
    return {"success": True, "data": data, "error": None}


def create_app(db_path: str | Path, service: LabelingService | None = None,
               proxies: ProxyManager | None = None, models: ModelRunner | None = None,
               dialog: FileDialog | None = None) -> FastAPI:
    svc = service or LabelingService(SegmentStore(db_path))
    proxy_mgr = proxies or ProxyManager(Path(db_path).parent / "labeler_proxies")
    runner = models or ModelRunner(ModelRegistry(DEFAULT_MODEL_DIR, DEFAULT_MODELS_ROOT))
    picker = dialog or FileDialog(Path.home() / "Downloads")
    app = FastAPI(title="VideoMAE Labeler")

    @app.middleware("http")
    async def _revalidate_static(request: Request, call_next):
        # Without this, browsers heuristically cache the ES modules and keep running
        # stale JS after an update. `no-cache` = always revalidate (cheap 304 via ETag).
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"success": False, "data": None, "error": exc.detail}, exc.status_code)

    def run(fn):
        try:
            return fn()
        except DialogBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (FileNotFoundError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface a readable error, log the detail
            log.exception("labeler request failed")
            raise HTTPException(500, f"伺服器錯誤：{exc}") from exc

    @app.get("/api/media")
    def media() -> dict:
        return _ok(run(svc.store.list_media))

    @app.post("/api/pick")
    def pick(body: PickBody) -> dict:
        """Show the native open-file dialog on this machine; returns {"path": str | None}."""
        return _ok(run(lambda: {"path": picker.pick_video(body.dir)}))

    @app.post("/api/open")
    def open_video(body: PathBody) -> dict:
        def opened():
            info = svc.open(body.path)
            return {"video": asdict(info), "segments": [asdict(s) for s in svc.segments(info.path)],
                    "model_available": runner.available}
        return _ok(run(opened))

    @app.post("/api/save")
    def save(body: SaveBody) -> dict:
        def saved():
            segments = [tl.Segment(**s.model_dump()) for s in body.segments]
            return {"segments": [asdict(s) for s in svc.save(body.path, segments)]}
        return _ok(run(saved))

    @app.get("/api/video")
    def video(path: str = Query(min_length=1, max_length=4096), proxy: bool = False) -> FileResponse:
        file = Path(path)
        if file.suffix.lower() not in VIDEO_EXTS or not file.is_file():
            raise HTTPException(404, f"找不到影片：{path}")
        if proxy:
            file = proxy_mgr.proxy_path(path)
            if not file.is_file():
                raise HTTPException(404, "相容播放檔尚未產生")
        return FileResponse(file, media_type="video/mp4" if proxy else None)

    @app.get("/api/proxy")
    def proxy_status(path: str = Query(min_length=1, max_length=4096)) -> dict:
        return _ok(run(lambda: asdict(proxy_mgr.status(svc.info(path).path))))

    @app.post("/api/proxy")
    def proxy_start(body: PathBody) -> dict:
        def started():
            info = svc.info(body.path)
            return asdict(proxy_mgr.start(info.path, info.frame_count / info.fps))
        return _ok(run(started))

    @app.get("/api/model")
    def model_status(path: str = Query(min_length=1, max_length=4096)) -> dict:
        return _ok(run(lambda: asdict(runner.status(svc.info(path).path))))

    @app.post("/api/model")
    def model_start(body: ModelRunBody) -> dict:
        return _ok(run(lambda: asdict(runner.start(svc.info(body.path).path, body.model))))

    @app.get("/api/models")
    def model_list() -> dict:
        return _ok(run(lambda: {"models": [asdict(m) for m in runner.registry.list()],
                                "default": runner.registry.default_id()}))

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
