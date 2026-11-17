"""Browser-playable H.264 proxies for videos the browser cannot decode.

OpenCV-written mp4s (MPEG-4 Part 2) and some camera codecs do not play in a
<video> element. The proxy keeps the source's frame rate and frame count
(`-fps_mode passthrough`), so frame numbers on the proxy match the original and
labels are still stored against the ORIGINAL video path.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyStatus:
    state: str = "none"          # none | running | ready | error
    progress: float = 0.0        # 0..1
    error: str | None = None


def proxy_command(ffmpeg: str, source: str, out: Path) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats", "-y", "-i", source,
            "-map", "0:v:0", "-map", "0:a:0?", "-fps_mode", "passthrough",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-g", "30", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            "-progress", "pipe:1", out.as_posix()]


def parse_progress(line: str, duration_s: float) -> float | None:
    """`out_time_us=12345678` -> fraction done, or None for other lines."""
    key, _, value = line.strip().partition("=")
    if key not in ("out_time_us", "out_time_ms") or not value.isdigit() or duration_s <= 0:
        return None
    return min(1.0, int(value) / 1e6 / duration_s)  # ffmpeg reports both keys in microseconds


class ProxyManager:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._status: dict[str, ProxyStatus] = {}
        self._lock = threading.Lock()

    def proxy_path(self, source: str) -> Path:
        src = Path(source)
        stat = src.stat()
        digest = hashlib.sha1(f"{src.resolve().as_posix()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
        return self.cache_dir / f"{src.stem[:40]}_{digest[:12]}.mp4"

    def status(self, source: str) -> ProxyStatus:
        with self._lock:
            current = self._status.get(source)
        if current is not None and current.state != "none":
            return current
        return ProxyStatus("ready", 1.0) if self.proxy_path(source).is_file() else ProxyStatus()

    def start(self, source: str, duration_s: float) -> ProxyStatus:
        current = self.status(source)
        if current.state in ("running", "ready"):
            return current
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            failed = ProxyStatus("error", 0.0, "找不到 ffmpeg，請先安裝並加入 PATH（例如 winget install Gyan.FFmpeg）")
            self._set(source, failed)
            return failed
        self._set(source, ProxyStatus("running", 0.0))
        threading.Thread(target=self._run, args=(ffmpeg, source, duration_s), daemon=True).start()
        return self.status(source)

    def _set(self, source: str, status: ProxyStatus) -> None:
        with self._lock:
            self._status = {**self._status, source: status}

    def _run(self, ffmpeg: str, source: str, duration_s: float) -> None:
        out = self.proxy_path(source)
        tmp = out.with_suffix(".part.mp4")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(proxy_command(ffmpeg, source, tmp), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                fraction = parse_progress(line, duration_s)
                if fraction is not None:
                    self._set(source, replace(self.status(source), progress=fraction))
            stderr = proc.stderr.read()
            if proc.wait() != 0:
                raise RuntimeError(stderr.strip()[-500:] or f"ffmpeg exit code {proc.returncode}")
            tmp.replace(out)
            self._set(source, ProxyStatus("ready", 1.0))
        except Exception as exc:  # noqa: BLE001 - report to the UI, keep the server alive
            log.exception("proxy transcode failed for %s", source)
            tmp.unlink(missing_ok=True)
            self._set(source, ProxyStatus("error", 0.0, f"轉檔失敗：{exc}"))
