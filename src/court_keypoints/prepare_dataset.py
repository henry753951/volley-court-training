
"""Export the 36-point court project and build an augmented YOLO pose dataset.

The first imported YOLO archive remains in the dataset but is deliberately not
used for offline augmentation. Database provenance is retained in manifest.json;
every in-frame point with visibility > 0 is emitted as ordinary ground truth,
including points whose source is ``predicted``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
import psycopg
import yaml
from dotenv import dotenv_values
from minio import Minio
from psycopg.rows import dict_row


LEGACY_BATCH = "volleyball-court-keypoints.v1i.yolov8.zip"
VIDEO_BATCH = "videoplayback.mp4"
HOMOGRAPHY_BATCH = "Volleyball Homography.v7i.yolov8.zip"
AUGMENT_BATCHES = {VIDEO_BATCH, HOMOGRAPHY_BATCH}


@dataclass
class Record:
    image_id: str
    file_name: str
    object_key: str
    width: int
    height: int
    sha256: str
    split: str
    status: str
    batch: str
    frame_index: int | None
    keypoints: list[dict[str, Any]]
    bbox: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-slug", default="volleyball-court-14-keypoints")
    parser.add_argument("--augment-copies", type=int, default=2)
    parser.add_argument(
        "--min-canonical-margin",
        type=int,
        default=1,
        help=(
            "Exclude labels whose best and second-best rectangle symmetry tie. "
            "A tied image cannot be assigned a trustworthy semantic direction."
        ),
    )
    parser.add_argument(
        "--min-canonical-ratio",
        type=float,
        default=0.90,
        help=(
            "Exclude labels whose canonical image-axis agreement is below this ratio. "
            "This catches arbitrary point-order mistakes that are not a rectangle symmetry."
        ),
    )
    parser.add_argument(
        "--grayscale-copies",
        type=int,
        choices=(0, 1),
        default=0,
        help="Write a parallel grayscale copy of every original image; originals are never replaced.",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--include-status",
        action="append",
        default=[],
        choices=("PENDING", "IN_PROGRESS", "COMPLETED", "REVIEWED"),
        help="Annotation image status to export; defaults to COMPLETED and REVIEWED.",
    )
    parser.add_argument(
        "--include-batch",
        action="append",
        default=[],
        help="Only export these source batch names; may be repeated.",
    )
    parser.add_argument(
        "--side-camera-canonicalize",
        action="store_true",
        help=(
            "Canonicalize the four rectangle symmetries. The scorer adapts between "
            "sideline views (court length is mostly horizontal) and endline views "
            "(court length is mostly vertical)."
        ),
    )
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def active_points(points: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index in range(36):
        point = points[index] if index < len(points) and points[index] else {}
        x, y = point.get("x"), point.get("y")
        visible = int(point.get("visibility") or 0) > 0
        in_frame = finite(x) and finite(y) and 0 <= float(x) < width and 0 <= float(y) < height
        normalized.append(
            {
                "index": index,
                "x": float(x) if in_frame else 0.0,
                "y": float(y) if in_frame else 0.0,
                "visibility": 2 if visible and in_frame else 0,
                "source": point.get("source") if visible and in_frame else None,
            }
        )
    return normalized


def derive_bbox(points: list[dict[str, Any]], width: int, height: int) -> tuple[float, float, float, float]:
    visible = [point for point in points if point["visibility"] > 0]
    xs = [point["x"] for point in visible]
    ys = [point["y"] for point in visible]
    pad_x = max(2.0, (max(xs) - min(xs)) * 0.03)
    pad_y = max(2.0, (max(ys) - min(ys)) * 0.03)
    x1 = max(0.0, min(xs) - pad_x)
    y1 = max(0.0, min(ys) - pad_y)
    x2 = min(float(width), max(xs) + pad_x)
    y2 = min(float(height), max(ys) + pad_y)
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, max(1.0, x2 - x1) / width, max(1.0, y2 - y1) / height)


def yolo_line(points: list[dict[str, Any]], width: int, height: int) -> str:
    bbox = derive_bbox(points, width, height)
    values = ["0", *(f"{value:.8f}" for value in bbox)]
    for point in points:
        if point["visibility"] > 0:
            values.extend((f"{point['x'] / width:.8f}", f"{point['y'] / height:.8f}", "2"))
        else:
            values.extend(("0", "0", "0"))
    return " ".join(values)


def source_alias(batch: str) -> str:
    return {LEGACY_BATCH: "legacy", VIDEO_BATCH: "video", HOMOGRAPHY_BATCH: "homography"}.get(batch, "other")


def load_project(database_url: str, slug: str) -> tuple[dict[str, Any], list[Record]]:
    query = '''
        SELECT
          p.id AS project_id, p.name AS project_name, p."className" AS class_name,
          p."keypointCount" AS keypoint_count, p."keypointSchema" AS keypoint_schema,
          p."flipIndex" AS flip_index, p.skeleton,
          i.id AS image_id, i."fileName" AS file_name, i."objectKey" AS object_key,
          i.width, i.height, i.sha256, i.split::text, i.status::text,
          a.keypoints, a.bbox,
          b."sourceFileName" AS batch, ia."frameIndex" AS frame_index
        FROM "Project" p
        JOIN "DatasetImage" i ON i."projectId" = p.id
        JOIN "Annotation" a ON a."imageId" = i.id
        LEFT JOIN "ImportAsset" ia ON ia."appliedImageId" = i.id
        LEFT JOIN "ImportBatch" b ON b.id = ia."importBatchId"
        WHERE p.slug = %s
        ORDER BY i."createdAt", i.id
    '''
    parsed = urlsplit(database_url)
    query_string = urlencode([(key, value) for key, value in parse_qsl(parsed.query) if key != "schema"])
    psycopg_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query_string, parsed.fragment))
    with psycopg.connect(psycopg_url, row_factory=dict_row) as connection:
        rows = connection.execute(query, (slug,)).fetchall()
    if not rows:
        raise RuntimeError(f"project has no annotated images: {slug}")
    first = rows[0]
    project = {
        "id": first["project_id"],
        "name": first["project_name"],
        "className": first["class_name"],
        "keypointCount": first["keypoint_count"],
        "keypointSchema": first["keypoint_schema"],
        "flipIndex": first["flip_index"],
        "skeleton": first["skeleton"],
    }
    records = [
        Record(
            image_id=row["image_id"], file_name=row["file_name"], object_key=row["object_key"],
            width=row["width"], height=row["height"], sha256=row["sha256"],
            split=row["split"], status=row["status"], batch=row["batch"] or "unknown",
            frame_index=row["frame_index"], keypoints=row["keypoints"], bbox=row["bbox"],
        )
        for row in rows
    ]
    return project, records


def video_groups(records: list[Record]) -> list[list[Record]]:
    ordered = sorted(records, key=lambda record: record.frame_index if record.frame_index is not None else 10**12)
    groups: list[list[Record]] = []
    for record in ordered:
        if not groups:
            groups.append([record])
            continue
        previous = groups[-1][-1].frame_index
        current = record.frame_index
        if previous is not None and current is not None and current - previous <= 300:
            groups[-1].append(record)
        else:
            groups.append([record])
    return groups


def assign_video_splits(records: list[Record], seed: int) -> dict[str, str]:
    targets = {"TRAIN": round(len(records) * 0.8), "VALID": round(len(records) * 0.1)}
    targets["TEST"] = len(records) - targets["TRAIN"] - targets["VALID"]
    assigned = Counter()
    result: dict[str, str] = {}
    groups = video_groups(records)
    groups.sort(key=lambda group: (-len(group), hashlib.sha256(f"{seed}:{group[0].image_id}".encode()).hexdigest()))
    for group in groups:
        split = max(targets, key=lambda name: (targets[name] - assigned[name]) / max(1, targets[name]))
        for record in group:
            result[record.image_id] = split
        assigned[split] += len(group)
    return result


def schema_coordinates(schema: list[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(schema, key=lambda point: int(point["index"]))
    coordinates = np.asarray(
        [(float(point["planeX"]), float(point["planeY"])) for point in ordered],
        dtype=np.float64,
    )
    if coordinates.shape != (36, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("36-point schema must provide finite planeX/planeY values")
    return coordinates


def symmetry_index(
    coordinates: np.ndarray,
    *,
    mirror_plane_x: bool,
    mirror_plane_y: bool,
) -> list[int]:
    result: list[int] = []
    for plane_x, plane_y in coordinates:
        target = np.asarray(
            [100.0 - plane_x if mirror_plane_x else plane_x, 100.0 - plane_y if mirror_plane_y else plane_y],
            dtype=np.float64,
        )
        distances = np.linalg.norm(coordinates - target, axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > 1e-4:
            raise RuntimeError(f"court schema has no symmetry pair for {target.tolist()}")
        result.append(index)
    return result


def reorder_points(points: list[dict[str, Any]], index_map: list[int]) -> list[dict[str, Any]]:
    reordered = [
        {"index": index, "x": 0.0, "y": 0.0, "visibility": 0, "source": None}
        for index in range(36)
    ]
    for source_index, point in enumerate(points):
        target_index = int(index_map[source_index])
        reordered[target_index] = {**point, "index": target_index}
    return reordered


def camera_view_mode(points: list[dict[str, Any]], coordinates: np.ndarray) -> str:
    """Classify whether the projected court length is mostly horizontal or vertical."""

    horizontal = 0.0
    vertical = 0.0
    for first in range(36):
        if not points[first]["visibility"]:
            continue
        for second in range(first + 1, 36):
            if not points[second]["visibility"]:
                continue
            plane_x_delta = coordinates[first, 0] - coordinates[second, 0]
            plane_y_delta = coordinates[first, 1] - coordinates[second, 1]
            if abs(plane_x_delta) < 1e-4 and abs(plane_y_delta) > 1e-4:
                horizontal += abs(points[first]["x"] - points[second]["x"])
                vertical += abs(points[first]["y"] - points[second]["y"])
    return "sideline" if horizontal >= vertical else "endline"


def side_camera_score(points: list[dict[str, Any]], coordinates: np.ndarray) -> tuple[int, int]:
    """Score image-axis monotonicity for upright sideline and endline cameras.

    Sideline convention: near-to-far moves image-left to image-right and
    physical left-to-right moves image-top to image-bottom.
    Endline convention: far-to-near moves image-top to image-bottom and
    physical left-to-right moves image-left to image-right.
    """

    view_mode = camera_view_mode(points, coordinates)
    good = 0
    bad = 0
    for first in range(36):
        if not points[first]["visibility"]:
            continue
        for second in range(first + 1, 36):
            if not points[second]["visibility"]:
                continue
            plane_x_delta = coordinates[first, 0] - coordinates[second, 0]
            plane_y_delta = coordinates[first, 1] - coordinates[second, 1]
            if abs(plane_x_delta) < 1e-4 and abs(plane_y_delta) > 1e-4:
                image_delta = (
                    points[first]["x"] - points[second]["x"]
                    if view_mode == "sideline"
                    else points[first]["y"] - points[second]["y"]
                )
                if abs(image_delta) > 1e-4:
                    desired_sign = (
                        -np.sign(plane_y_delta)
                        if view_mode == "sideline"
                        else np.sign(plane_y_delta)
                    )
                    if np.sign(image_delta) == desired_sign:
                        good += 1
                    else:
                        bad += 1
            if abs(plane_y_delta) < 1e-4 and abs(plane_x_delta) > 1e-4:
                image_delta = (
                    points[first]["y"] - points[second]["y"]
                    if view_mode == "sideline"
                    else points[first]["x"] - points[second]["x"]
                )
                if abs(image_delta) > 1e-4:
                    if np.sign(image_delta) == np.sign(plane_x_delta):
                        good += 1
                    else:
                        bad += 1
    return good - bad, good + bad


def canonicalize_side_camera(
    points: list[dict[str, Any]],
    coordinates: np.ndarray,
    symmetries: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[int, int, str, list[dict[str, Any]]]] = []
    for name, index_map in symmetries.items():
        candidate = reorder_points(points, index_map)
        score, comparisons = side_camera_score(candidate, coordinates)
        candidates.append((score, comparisons, name, candidate))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, comparisons, transform, canonical = candidates[0]
    second_score = candidates[1][0]
    return canonical, {
        "transform": transform,
        "viewMode": camera_view_mode(canonical, coordinates),
        "score": best_score,
        "comparisons": comparisons,
        "margin": best_score - second_score,
        "candidateScores": {name: score for score, _pairs, name, _points in candidates},
    }


def minio_client(config: dict[str, str | None]) -> tuple[Minio, str]:
    endpoint = str(config["MINIO_ENDPOINT"])
    port = int(str(config.get("MINIO_PORT") or 9000))
    return (
        Minio(
            f"{endpoint}:{port}",
            access_key=str(config["MINIO_ACCESS_KEY"]),
            secret_key=str(config["MINIO_SECRET_KEY"]),
            secure=str(config.get("MINIO_USE_SSL") or "false").lower() == "true",
        ),
        str(config["MINIO_BUCKET"]),
    )


def get_bytes(client: Minio, bucket: str, object_key: str) -> bytes:
    response = client.get_object(bucket, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def transformed_copy(image: np.ndarray, points: list[dict[str, Any]], flip_index: list[int], seed: int, force_flip: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    output = image.copy()
    transformed = [dict(point) for point in points]
    if force_flip:
        output = cv2.flip(output, 1)
        mirrored = [{"index": index, "x": 0.0, "y": 0.0, "visibility": 0, "source": None} for index in range(36)]
        for source_index, point in enumerate(transformed):
            target_index = int(flip_index[source_index])
            mirrored[target_index] = {**point, "index": target_index, "x": width - 1 - point["x"] if point["visibility"] else 0.0}
        transformed = mirrored
    angle = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.96, 1.04))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += float(rng.uniform(-0.025, 0.025) * width)
    matrix[1, 2] += float(rng.uniform(-0.025, 0.025) * height)
    output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    for point in transformed:
        if not point["visibility"]:
            continue
        x, y = matrix @ np.array([point["x"], point["y"], 1.0])
        if 0 <= x < width and 0 <= y < height:
            point["x"], point["y"] = float(x), float(y)
        else:
            point.update(x=0.0, y=0.0, visibility=0, source=None)
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.82, 1.18)
    hsv[..., 2] *= rng.uniform(0.82, 1.18)
    output = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    gamma = float(rng.uniform(0.82, 1.18))
    lookup = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8)
    output = cv2.LUT(output, lookup)
    if rng.random() < 0.25:
        output = cv2.GaussianBlur(output, (3, 3), float(rng.uniform(0.2, 0.8)))
    return output, transformed


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    config = {**dotenv_values(args.annotator_env), **{key: value for key, value in os.environ.items() if key.startswith(("DATABASE_", "MINIO_"))}}
    required = ["DATABASE_URL", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise RuntimeError(f"missing annotator environment keys: {', '.join(missing)}")
    project, records = load_project(str(config["DATABASE_URL"]), args.project_slug)
    if project["keypointCount"] != 36:
        raise RuntimeError(f"expected 36 keypoints, got {project['keypointCount']}")
    included_statuses = set(args.include_status or ("COMPLETED", "REVIEWED"))
    records = [record for record in records if record.status in included_statuses]
    if not records:
        raise RuntimeError(f"no records matched statuses: {sorted(included_statuses)}")
    if args.include_batch:
        included_batches = set(args.include_batch)
        records = [record for record in records if record.batch in included_batches]
        if not records:
            raise RuntimeError(f"no records matched --include-batch: {sorted(included_batches)}")
    schema = sorted(project["keypointSchema"], key=lambda point: point["index"])
    coordinates = schema_coordinates(schema)
    symmetries = {
        "identity": symmetry_index(coordinates, mirror_plane_x=False, mirror_plane_y=False),
        "width": symmetry_index(coordinates, mirror_plane_x=True, mirror_plane_y=False),
        "length": symmetry_index(coordinates, mirror_plane_x=False, mirror_plane_y=True),
        "both": symmetry_index(coordinates, mirror_plane_x=True, mirror_plane_y=True),
    }
    # A horizontal image flip reverses different physical axes by view:
    # sideline -> court length, endline -> court width. Ultralytics accepts only
    # one global flip_idx, so adaptive datasets disable its built-in fliplr and
    # generate correctly permuted flip augmentations here per image.
    per_view_flip_index = {
        "sideline": symmetries["length"],
        "endline": symmetries["width"],
    }
    effective_flip_index = (
        symmetries["identity"] if args.side_camera_canonicalize else project["flipIndex"]
    )
    client, bucket = minio_client(config)
    video_split = assign_video_splits([record for record in records if record.batch == VIDEO_BATCH], args.seed)
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "version": 2,
        "project": project,
        "selection": {
            "includeBatches": args.include_batch or None,
            "includeStatuses": sorted(included_statuses),
            "minimumCanonicalMargin": args.min_canonical_margin,
            "minimumCanonicalRatio": args.min_canonical_ratio,
        },
        "canonicalization": {
            "adaptiveCamera": args.side_camera_canonicalize,
            "supportedViews": ["sideline", "endline"],
            "datasetFlipIndex": effective_flip_index,
            "perViewHorizontalFlipIndex": per_view_flip_index,
            "requiresBuiltInFliplrZero": args.side_camera_canonicalize,
        },
        "augmentation": {"excludedBatch": LEGACY_BATCH, "copies": args.augment_copies, "seed": args.seed},
        "grayscale": {
            "copies": args.grayscale_copies,
            "layout": "<split>-grayscale/images + <split>-grayscale/labels",
            "originalsPreserved": True,
        },
        "images": [],
        "excluded": [],
    }
    direction_rows: list[dict[str, Any]] = []
    original_counts: Counter[tuple[str, str]] = Counter()
    augmented_counts: Counter[str] = Counter()
    for number, record in enumerate(records, start=1):
        points = active_points(record.keypoints, record.width, record.height)
        orientation: dict[str, Any] | None = None
        if args.side_camera_canonicalize:
            points, orientation = canonicalize_side_camera(points, coordinates, symmetries)
            if orientation["margin"] < args.min_canonical_margin:
                reason = (
                    "ambiguous rectangle symmetry: "
                    f"margin {orientation['margin']} < {args.min_canonical_margin}"
                )
                manifest["excluded"].append({
                    "id": record.image_id,
                    "file": record.file_name,
                    "reason": reason,
                    "batch": record.batch,
                    "status": record.status,
                    "orientation": orientation,
                })
                direction_rows.append({
                    "imageId": record.image_id,
                    "file": record.file_name,
                    "batch": record.batch,
                    "split": record.split,
                    "visible": sum(point["visibility"] > 0 for point in points),
                    "transform": orientation["transform"],
                    "viewMode": orientation["viewMode"],
                    "score": orientation["score"],
                    "comparisons": orientation["comparisons"],
                    "margin": orientation["margin"],
                    "scoreRatio": (
                        orientation["score"] / orientation["comparisons"]
                        if orientation["comparisons"]
                        else -1.0
                    ),
                    "included": False,
                    "reason": reason,
                })
                continue
            score_ratio = (
                orientation["score"] / orientation["comparisons"]
                if orientation["comparisons"]
                else -1.0
            )
            if score_ratio < args.min_canonical_ratio:
                reason = (
                    "non-canonical point order: "
                    f"score ratio {score_ratio:.4f} < {args.min_canonical_ratio:.4f}"
                )
                manifest["excluded"].append({
                    "id": record.image_id,
                    "file": record.file_name,
                    "reason": reason,
                    "batch": record.batch,
                    "status": record.status,
                    "orientation": orientation,
                })
                direction_rows.append({
                    "imageId": record.image_id,
                    "file": record.file_name,
                    "batch": record.batch,
                    "split": record.split,
                    "visible": sum(point["visibility"] > 0 for point in points),
                    "transform": orientation["transform"],
                    "viewMode": orientation["viewMode"],
                    "score": orientation["score"],
                    "comparisons": orientation["comparisons"],
                    "margin": orientation["margin"],
                    "scoreRatio": score_ratio,
                    "included": False,
                    "reason": reason,
                })
                continue
        visible_count = sum(point["visibility"] > 0 for point in points)
        if visible_count == 0:
            manifest["excluded"].append({"id": record.image_id, "file": record.file_name, "reason": "zero in-frame keypoints", "batch": record.batch, "status": record.status})
            continue
        split_name = video_split.get(record.image_id, record.split)
        if split_name not in {"TRAIN", "VALID", "TEST"}:
            raise RuntimeError(f"unassigned split for {record.image_id}")
        split_dir = {"TRAIN": "train", "VALID": "valid", "TEST": "test"}[split_name]
        alias = source_alias(record.batch)
        base_name = f"{alias}__{record.image_id}"
        image_ext = Path(record.file_name).suffix.lower() or ".jpg"
        image_bytes = get_bytes(client, bucket, record.object_key)
        image_path = output / split_dir / "images" / f"{base_name}{image_ext}"
        label_path = output / split_dir / "labels" / f"{base_name}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        label_text = yolo_line(points, record.width, record.height) + "\n"
        label_path.write_text(label_text, encoding="utf-8")
        grayscale_file: str | None = None
        if args.grayscale_copies:
            decoded_original = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded_original is None:
                raise RuntimeError(f"OpenCV could not decode {record.file_name}")
            grayscale = cv2.cvtColor(decoded_original, cv2.COLOR_BGR2GRAY)
            grayscale_image_path = output / f"{split_dir}-grayscale" / "images" / f"{base_name}.jpg"
            grayscale_label_path = output / f"{split_dir}-grayscale" / "labels" / f"{base_name}.txt"
            grayscale_image_path.parent.mkdir(parents=True, exist_ok=True)
            grayscale_label_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(grayscale_image_path), grayscale, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write grayscale image: {grayscale_image_path}")
            grayscale_label_path.write_text(label_text, encoding="utf-8")
            grayscale_file = str(grayscale_image_path.relative_to(output)).replace("\\", "/")
        canonical_score, canonical_comparisons = side_camera_score(points, coordinates)
        audit = orientation or {
            "transform": "none",
            "viewMode": camera_view_mode(points, coordinates),
            "score": canonical_score,
            "comparisons": canonical_comparisons,
            "margin": 0,
            "candidateScores": {},
        }
        direction_rows.append(
            {
                "imageId": record.image_id,
                "file": record.file_name,
                "batch": record.batch,
                "split": split_name,
                "visible": visible_count,
                "transform": audit["transform"],
                "viewMode": audit["viewMode"],
                "score": audit["score"],
                "comparisons": audit["comparisons"],
                "margin": audit["margin"],
                "scoreRatio": (
                    audit["score"] / audit["comparisons"]
                    if audit["comparisons"]
                    else -1.0
                ),
                "included": True,
                "reason": "",
            }
        )
        sources = Counter(point["source"] or "unknown" for point in points if point["visibility"])
        manifest["images"].append({"id": record.image_id, "file": str(image_path.relative_to(output)).replace("\\", "/"), "grayscaleFile": grayscale_file, "label": str(label_path.relative_to(output)).replace("\\", "/"), "originalFile": record.file_name, "sha256": record.sha256, "batch": record.batch, "split": split_name, "status": record.status, "frameIndex": record.frame_index, "visibleKeypoints": visible_count, "pointSources": dict(sources), "orientation": audit, "augmented": []})
        original_counts[(record.batch, split_name)] += 1
        if split_name == "TRAIN" and record.batch in AUGMENT_BATCHES and args.augment_copies > 0:
            decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"OpenCV could not decode {record.file_name}")
            for copy_index in range(args.augment_copies):
                aug_seed = int(hashlib.sha256(f"{args.seed}:{record.image_id}:{copy_index}".encode()).hexdigest()[:16], 16)
                augmentation_flip_index = (
                    per_view_flip_index[audit["viewMode"]]
                    if args.side_camera_canonicalize
                    else effective_flip_index
                )
                aug_image, aug_points = transformed_copy(decoded, points, augmentation_flip_index, aug_seed, force_flip=copy_index % 2 == 1)
                aug_name = f"{base_name}__aug{copy_index + 1}"
                aug_image_path = output / "train" / "images" / f"{aug_name}.jpg"
                aug_label_path = output / "train" / "labels" / f"{aug_name}.txt"
                cv2.imwrite(str(aug_image_path), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                aug_label_path.write_text(yolo_line(aug_points, record.width, record.height) + "\n", encoding="utf-8")
                manifest["images"][-1]["augmented"].append(str(aug_image_path.relative_to(output)).replace("\\", "/"))
                augmented_counts[record.batch] += 1
        if number % 50 == 0:
            print(f"exported {number}/{len(records)}")
    dataset_common = {"path": str(output).replace("\\", "/"), "kpt_shape": [36, 3], "flip_idx": effective_flip_index, "kpt_names": [point["name"] for point in schema], "nc": 1, "names": [project["className"]]}
    color_yaml = {**dataset_common, "train": "train/images", "val": "valid/images", "test": "test/images"}
    grayscale_yaml = {**dataset_common, "train": "train-grayscale/images", "val": "valid-grayscale/images", "test": "test-grayscale/images"}
    mixed_yaml = {**dataset_common, "train": ["train/images", "train-grayscale/images"], "val": "valid/images", "test": "test/images"}
    (output / "dataset-color.yaml").write_text(yaml.safe_dump(color_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.grayscale_copies:
        (output / "dataset-grayscale.yaml").write_text(yaml.safe_dump(grayscale_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (output / "dataset.yaml").write_text(yaml.safe_dump(mixed_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        (output / "dataset.yaml").write_text(yaml.safe_dump(color_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest["summary"] = {"databaseRows": len(records), "usableOriginals": len(manifest["images"]), "grayscaleOriginals": sum(item["grayscaleFile"] is not None for item in manifest["images"]), "excluded": len(manifest["excluded"]), "originalByBatchSplit": {f"{batch}|{split}": count for (batch, split), count in sorted(original_counts.items())}, "augmentedByBatch": dict(augmented_counts), "totalTrainImages": len(list((output / "train" / "images").glob("*"))), "totalValidImages": len(list((output / "valid" / "images").glob("*"))), "totalTestImages": len(list((output / "test" / "images").glob("*"))), "totalGrayscaleTrainImages": len(list((output / "train-grayscale" / "images").glob("*"))), "totalGrayscaleValidImages": len(list((output / "valid-grayscale" / "images").glob("*"))), "totalGrayscaleTestImages": len(list((output / "test-grayscale" / "images").glob("*")))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "orientation-audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_rows[0]))
        writer.writeheader()
        writer.writerows(direction_rows)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
