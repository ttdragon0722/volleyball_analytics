"""Background VideoMAE game-state pre-labelling, reusing game_state_clipper.

Results are returned to the browser as *model* segments (play / service only);
they are NOT written to the DB here. The browser merges them under the user's
own labels and everything is saved together on Ctrl+S.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .model_registry import ModelRegistry
from .timeline import MODEL, Segment

log = logging.getLogger(__name__)
ACTIVE_STATES = frozenset({"service", "play"})
SMOOTH_K = 3

# (video_path, model_dir, on_progress(fraction)) -> windows with start_frame, end_frame (exclusive), state, confidence
ClassifyFn = Callable[[str, str, Callable[[float], None]], Sequence]


@dataclass(frozen=True)
class JobStatus:
    state: str = "none"                  # none | running | ready | error
    progress: float = 0.0
    error: str | None = None
    segments: tuple[Segment, ...] = field(default=())
    model: str | None = None             # model folder used for this job


def windows_to_segments(windows: Sequence, smooth_k: int = SMOOTH_K) -> tuple[Segment, ...]:
    """Smooth window states, merge consecutive equal states, keep only play/service."""
    from game_state_clipper.segments import smooth_states

    states = smooth_states([w.state for w in windows], smooth_k) if windows else []
    runs: list[tuple[int, int, str, list[float]]] = []
    for win, state in zip(windows, states):
        if runs and runs[-1][2] == state and runs[-1][1] + 1 == win.start_frame:
            start, _, _, confs = runs[-1]
            runs[-1] = (start, win.end_frame - 1, state, [*confs, win.confidence])
        else:
            runs.append((win.start_frame, win.end_frame - 1, state, [win.confidence]))
    return tuple(
        Segment(start, end, state, MODEL, round(min(1.0, max(0.0, sum(c) / len(c))), 4))
        for start, end, state, c in runs
        if state in ACTIVE_STATES and end >= start)


def default_classify_fn() -> ClassifyFn:
    """Real classifier. Keeps ONE model in GPU memory; switching models unloads the previous one."""
    lock = threading.Lock()
    loaded: dict[str, object] = {}

    def classify(path: str, model_dir: str, on_progress: Callable[[float], None]) -> Sequence:
        from game_state_clipper.classify import ClassifyConfig, classify_video, load_classifier, probe_video

        with lock:
            if loaded.get("dir") != model_dir:
                loaded.clear()
                _free_gpu()
                loaded["clf"] = load_classifier(Path(model_dir))
                loaded["dir"] = model_dir
            clf = loaded["clf"]
            info = probe_video(path)
            done = [0]

            def progress(n_frames: int, _window) -> None:
                done[0] += n_frames
                on_progress(min(1.0, done[0] / max(1, info.total_frames)))

            return classify_video(info, clf, ClassifyConfig(), progress)

    return classify


def _free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class ModelRunner:
    def __init__(self, registry: ModelRegistry, classify_fn: ClassifyFn | None = None) -> None:
        self.registry = registry
        self._classify = classify_fn or default_classify_fn()
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.registry.default_id() is not None

    def status(self, path: str) -> JobStatus:
        with self._lock:
            return self._jobs.get(path, JobStatus())

    def start(self, path: str, model_id: str | None = None) -> JobStatus:
        """Pre-label `path` with the chosen model (None = registry default). Unknown ids raise LookupError."""
        try:
            model_dir = self.registry.resolve(model_id)
        except FileNotFoundError as exc:
            return JobStatus("error", 0.0, str(exc))
        with self._lock:
            if self._jobs.get(path, JobStatus()).state == "running":
                return self._jobs[path]
            self._jobs = {**self._jobs, path: JobStatus("running", model=model_dir)}
        threading.Thread(target=self._run, args=(path, model_dir), daemon=True).start()
        return self.status(path)

    def _set(self, path: str, status: JobStatus) -> None:
        with self._lock:
            self._jobs = {**self._jobs, path: status}

    def _run(self, path: str, model_dir: str) -> None:
        try:
            windows = self._classify(path, model_dir,
                                     lambda f: self._set(path, JobStatus("running", f, model=model_dir)))
            self._set(path, JobStatus("ready", 1.0, None, windows_to_segments(windows), model_dir))
        except Exception as exc:  # noqa: BLE001 - report to the UI, keep the server alive
            log.exception("model pre-labelling failed for %s", path)
            self._set(path, JobStatus("error", 0.0, f"模型預標失敗：{exc}", model=model_dir))
