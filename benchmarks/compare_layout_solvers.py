from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFile

from volley_court import CourtLineModel
from volley_court.evaluate_layout import evaluate
from volley_court.layout import match_court_layout, match_semantic_court_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare search and fixed semantic layout solvers")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--solvers", nargs="+", choices=("search", "semantic"), default=("search", "semantic")
    )
    parser.add_argument("--write-overlays", action="store_true")
    return parser.parse_args()


def _test_images(dataset_yaml: Path) -> list[Path]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = dataset_yaml.parent / str(payload.get("path", "."))
    values = payload["test"]
    directories = values if isinstance(values, list) else [values]
    return sorted(
        path
        for value in directories
        for path in (root / str(value)).resolve().iterdir()
        if path.is_file()
    )


def _read_image(path: Path) -> np.ndarray:
    for _attempt in range(3):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))[:, :, ::-1].copy()


def _predict(
    checkpoint: Path,
    images: list[Path],
    output: Path,
    *,
    device: str,
    batch_size: int,
    confidence: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model = CourtLineModel.from_pretrained(
        checkpoint, device=device, decoder="cuda", confidence=confidence
    )
    model.warmup(batch_size=batch_size)
    for start in range(0, len(images), batch_size):
        batch_paths = images[start : start + batch_size]
        frames = [_read_image(path) for path in batch_paths]
        results = model.predict_many(frames, include_layout=False)
        for path, result in zip(batch_paths, results, strict=True):
            payload = {"segments": [line.to_mapping() for line in result.lines]}
            (output / f"{path.stem}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        del frames, results
        gc.collect()


def _runtime(
    images: list[Path],
    predictions: Path,
    *,
    repeats: int,
    solvers: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    cases = []
    for image_path in images:
        image = _read_image(image_path)
        payload = json.loads((predictions / f"{image_path.stem}.json").read_text(encoding="utf-8"))
        cases.append((payload["segments"], image.shape[1], image.shape[0]))
    output = {}
    available = {"search": match_court_layout, "semantic": match_semantic_court_layout}
    for name in solvers:
        solver = available[name]
        values = []
        for _repeat in range(repeats):
            for segments, width, height in cases:
                start = time.perf_counter_ns()
                solver(segments, width, height)
                values.append((time.perf_counter_ns() - start) / 1e6)
        output[name] = {
            "samples": float(len(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)),
        }
    return output


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    topology = args.topology.resolve()
    output = args.output.resolve()
    predictions = output / "predictions"
    images = _test_images(dataset)
    _predict(
        args.checkpoint.resolve(),
        images,
        predictions,
        device=args.device,
        batch_size=args.batch_size,
        confidence=args.confidence,
    )
    reports: dict[str, Any] = {}
    solvers = tuple(dict.fromkeys(args.solvers))
    for solver in solvers:
        reports[solver] = evaluate(
            dataset,
            predictions,
            topology,
            output / solver,
            split="test",
            layout_solver=solver,
            write_overlays=args.write_overlays,
        )
    comparison = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(dataset),
        "images": len(images),
        "runtime": _runtime(images, predictions, repeats=args.repeats, solvers=solvers),
        "quality": reports,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
