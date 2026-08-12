from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import numpy as np

from .types import Image


class VideoWriter(Protocol):
    codec: str

    def write(self, frame: Image) -> None: ...

    def close(self) -> None: ...


class WebVideoWriter:
    """Write browser-compatible H.264 MP4 with optional source audio."""

    codec = "h264/yuv420p/faststart"

    def __init__(
        self,
        output: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        audio_source: str | Path | None = None,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for browser-compatible MP4 output")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
        ]
        if audio_source is not None:
            command.extend(
                [
                    "-i",
                    str(audio_source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                ]
            )
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if audio_source is not None:
            command.extend(["-c:a", "aac", "-b:a", "160k", "-shortest"])
        command.extend(["-movflags", "+faststart", str(output)])
        self._width = width
        self._height = height
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg did not open its raw-video input pipe")
        self._stdin = cast(BinaryIO, self._process.stdin)
        self._closed = False

    def write(self, frame: Image) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")
        if frame.shape != (self._height, self._width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 BGR frame {(self._height, self._width, 3)}, got "
                f"{frame.shape} {frame.dtype}"
            )
        self._stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stdin.close()
        return_code = self._process.wait()
        stderr = (
            self._process.stderr.read().decode("utf-8", errors="replace")
            if self._process.stderr is not None
            else ""
        )
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")


def open_video_writer(
    output: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
    audio_source: str | Path | None = None,
) -> VideoWriter:
    return WebVideoWriter(
        output,
        width=width,
        height=height,
        fps=fps,
        audio_source=audio_source,
    )
