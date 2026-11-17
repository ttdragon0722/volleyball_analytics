"""Native "open file" dialog for picking a video.

A browser <input type=file> never exposes the full path, but labels are keyed
by the video's absolute path, so the (local) server shows the OS dialog itself.
The dialog runs in a separate Python process so Tk never touches the server's
worker threads.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from .service import VIDEO_EXTS

_DIALOG_SCRIPT = r"""
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
exts = " ".join("*" + e for e in sys.argv[2].split(","))
path = filedialog.askopenfilename(
    parent=root, title="選擇要標註的影片", initialdir=sys.argv[1] or None,
    filetypes=[("影片", exts), ("所有檔案", "*.*")])
root.destroy()
sys.stdout.write(path or "")
"""


class DialogBusyError(RuntimeError):
    pass


class FileDialog:
    def __init__(self, default_dir: str | Path) -> None:
        self.default_dir = Path(default_dir)
        self._lock = threading.Lock()

    def _initial_dir(self, requested: str) -> str:
        for candidate in (requested, str(self.default_dir)):
            if candidate and Path(candidate).is_dir():
                return Path(candidate).as_posix()
        return ""

    def pick_video(self, initial_dir: str = "") -> str | None:
        """Block until the user picks a video (absolute posix path) or cancels (None)."""
        if not self._lock.acquire(blocking=False):
            raise DialogBusyError("已經有一個選擇檔案的視窗開著，請先在那個視窗選擇或取消")
        try:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [sys.executable, "-I", "-c", _DIALOG_SCRIPT, self._initial_dir(initial_dir),
                 ",".join(sorted(VIDEO_EXTS))],
                capture_output=True, text=True, encoding="utf-8", env=env, check=False)
        finally:
            self._lock.release()
        if result.returncode != 0:
            raise RuntimeError(f"無法開啟選擇檔案視窗：{result.stderr.strip()[-300:]}")
        path = result.stdout.strip()
        return Path(path).resolve().as_posix() if path else None
