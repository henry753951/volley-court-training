"""Download the fixed YouTube evaluation set at <=720p and maximum available FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval-videos.yaml")
    parser.add_argument("--force", action="store_true", help="Allow yt-dlp to overwrite completed files.")
    return parser.parse_args()


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    if int(stream["height"]) > 720:
        raise RuntimeError(f"downloaded video exceeds 720p: {path} ({stream['height']}p)")
    return payload


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", []) if isinstance(payload, dict) else []
    if not videos:
        raise RuntimeError(f"evaluation manifest has no videos: {manifest_path}")

    reports: list[dict[str, Any]] = []
    for item in videos:
        video_id = str(item["id"])
        target = (manifest_path.parent / str(item["file"])).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--extractor-args",
            "youtube:player_client=android_vr",
            "--impersonate",
            "chrome",
            "--format",
            "bv*[height<=720]+ba/b[height<=720]",
            "--format-sort",
            "res:720,fps,vcodec:h264,acodec:aac",
            "--merge-output-format",
            "mp4",
            "--concurrent-fragments",
            "4",
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--write-info-json",
            "--output",
            str(target),
        ]
        command.append("--force-overwrites" if args.force else "--no-overwrites")
        command.append(str(item["url"]))
        print(f"downloading {video_id} -> {target}")
        subprocess.run(command, check=True)
        if not target.is_file():
            raise FileNotFoundError(f"yt-dlp did not create expected file: {target}")
        reports.append(
            {
                "id": video_id,
                "url": str(item["url"]),
                "file": str(target),
                "selection": {
                    "maxHeight": 720,
                    "sort": ["resolution=720", "fps=highest", "h264 preferred", "aac preferred"],
                    "youtubePlayerClient": "android_vr",
                    "impersonation": "chrome",
                },
                "probe": probe_video(target),
            }
        )

    report_path = manifest_path.parent / "artifacts" / "eval-videos" / "download-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"manifest": str(manifest_path), "videos": reports}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
