
"""Train the YOLO26 pose baseline on the prepared court keypoint dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_evaluation_videos(manifest_path: Path) -> list[dict[str, str]]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", []) if isinstance(payload, dict) else []
    if not videos:
        raise RuntimeError(f"evaluation manifest has no videos: {manifest_path}")
    resolved: list[dict[str, str]] = []
    for item in videos:
        if not isinstance(item, dict) or not item.get("id") or not item.get("file"):
            raise RuntimeError(f"invalid evaluation video entry: {item!r}")
        video_path = (manifest_path.parent / str(item["file"])).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(
                f"evaluation video is missing: {video_path}; run court-download-eval-videos first"
            )
        resolved.append({"id": str(item["id"]), "file": str(video_path)})
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume optimizer, scheduler, scaler, EMA, and epoch state from an Ultralytics last.pt checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=float, default=2, help="Batch size; use 0.9 for 90%% automatic GPU memory use")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--name", default="court36-color-distortion-v1")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.0,
        help="Keep at 0 for adaptive sideline/endline datasets; flips are precomputed per view.",
    )
    parser.add_argument("--hsv-h", type=float, default=0.03)
    parser.add_argument("--hsv-s", type=float, default=0.50)
    parser.add_argument("--hsv-v", type=float, default=0.40)
    parser.add_argument("--degrees", type=float, default=3.0)
    parser.add_argument("--translate", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--perspective", type=float, default=0.0005)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--eval-videos",
        type=Path,
        default=PROJECT_ROOT / "eval-videos.yaml",
        help="Fixed evaluation manifest. Every successful training run evaluates every listed video.",
    )
    parser.add_argument("--eval-imgsz", type=int, default=1280)
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--eval-point-conf", type=float, default=0.25)
    args = parser.parse_args()

    evaluation_manifest = args.eval_videos.resolve()
    evaluation_videos = load_evaluation_videos(evaluation_manifest)
    dataset_manifest = args.data.resolve().parent / "manifest.json"
    if dataset_manifest.is_file():
        manifest_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        canonicalization = manifest_payload.get("canonicalization", {})
        if canonicalization.get("requiresBuiltInFliplrZero") and args.fliplr != 0:
            raise ValueError(
                "adaptive sideline/endline data requires --fliplr 0 because one global "
                "Ultralytics flip_idx cannot represent both camera regimes"
            )

    from ultralytics import YOLO

    resume_checkpoint = args.resume.resolve() if args.resume else None
    if resume_checkpoint and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {resume_checkpoint}")
    model = YOLO(str(resume_checkpoint or args.model))
    batch = int(args.batch) if args.batch >= 1 and args.batch.is_integer() else args.batch
    cache: bool | str = False if args.cache == "false" else args.cache
    train_options = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        cache=cache,
        optimizer=args.optimizer,
        seed=args.seed,
        amp=True,
        cos_lr=args.cos_lr,
        multi_scale=args.multi_scale,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        compile=args.compile,
        plots=True,
        save=True,
    )
    if resume_checkpoint:
        train_options.update(
            resume=str(resume_checkpoint),
            save_dir=str(args.project.resolve() / args.name),
        )
    results = model.train(**train_options)
    save_dir = Path(results.save_dir)
    print(f"save_dir={save_dir}")
    print(f"best={save_dir / 'weights' / 'best.pt'}")
    print(f"last={save_dir / 'weights' / 'last.pt'}")
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.is_file():
        best_model = save_dir / "weights" / "last.pt"
    evaluation_root = save_dir / "evaluations"
    evaluation_rows: list[dict[str, str]] = []
    for video in evaluation_videos:
        output_dir = evaluation_root / video["id"]
        command = [
            sys.executable,
            "-m",
            "court_keypoints.evaluate_video",
            "--model",
            str(best_model),
            "--video",
            video["file"],
            "--output",
            str(output_dir),
            "--imgsz",
            str(args.eval_imgsz),
            "--device",
            str(args.device),
            "--conf",
            str(args.eval_conf),
            "--point-conf",
            str(args.eval_point_conf),
        ]
        print(f"evaluating {video['id']}: {video['file']}")
        subprocess.run(command, check=True)
        evaluation_rows.append(
            {
                "id": video["id"],
                "video": video["file"],
                "output": str(output_dir),
                "overlay": str(output_dir / "overlay.mp4"),
                "contactSheet": str(output_dir / "contact-sheet.jpg"),
                "metrics": str(output_dir / "metrics.json"),
            }
        )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "index.json").write_text(
        json.dumps(
            {
                "manifest": str(evaluation_manifest),
                "model": str(best_model),
                "videos": evaluation_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
