"""Cut segments out of the source video with ffmpeg and write the reports."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from .segments import Segment, Window


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FileNotFoundError("找不到 ffmpeg，請先安裝並加入 PATH（例如 winget install Gyan.FFmpeg）")
    return exe


def clip_name(index: int, segment: Segment) -> str:
    def stamp(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"
    states = "+".join(segment.states)
    return f"clip_{index:03d}_{stamp(segment.start_s)}-{stamp(segment.end_s)}_{states}.mp4"


def ffmpeg_command(ffmpeg: str, video: str, segment: Segment, out: Path, copy: bool) -> list[str]:
    # -ss before -i seeks fast; with re-encoding the cut is still frame-accurate.
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{segment.start_s:.3f}", "-i", video, "-t", f"{segment.duration_s:.3f}"]
    if copy:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "128k"]
    return cmd + [out.as_posix()]


def cut_segment(ffmpeg: str, video: str, segment: Segment, out: Path, copy: bool) -> None:
    proc = subprocess.run(ffmpeg_command(ffmpeg, video, segment, out, copy),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 剪輯失敗 ({out.name}): {proc.stderr.strip()[-500:]}")


def write_reports(out_dir: Path, video: str, fps: float, windows: list[Window],
                  segments: list[Segment], clip_names: list[str]) -> None:
    timeline = {
        "video": video,
        "fps": fps,
        "windows": [{**asdict(w), "start_s": round(w.start_frame / fps, 3)} for w in windows],
    }
    (out_dir / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(out_dir / "segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "start_s", "end_s", "duration_s", "states"])
        for name, seg in zip(clip_names, segments):
            writer.writerow([name, f"{seg.start_s:.2f}", f"{seg.end_s:.2f}", f"{seg.duration_s:.2f}",
                             "+".join(seg.states)])
