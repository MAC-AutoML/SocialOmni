from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def create_black_frame_video(input_path: str, output_dir: str) -> str:
    """Create a video with black frames while preserving the original audio track."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot create masked-video input")

    source = Path(input_path)
    target = Path(output_dir) / f"{source.stem}_masked_video.mp4"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "drawbox=x=0:y=0:w=iw:h=ih:color=black@1:t=fill,format=yuv420p",
        "-c:v",
        "mpeg4",
        "-q:v",
        "31",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg masked-video generation failed: {completed.stderr[-1000:]}")
    return str(target)
