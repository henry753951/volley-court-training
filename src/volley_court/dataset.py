from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import PointSample, clip_segment_to_image, parse_yolo_pose_line
from .layout import CANONICAL_KEYPOINTS

LAYOUT_CORNER_IDS = (0, 4, 5, 9)
LAYOUT_SYMMETRY_PERMUTATIONS = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
)


def _pose36_symmetry_maps() -> tuple[tuple[int, ...], ...]:
    coordinate_to_index = {
        coordinate: index for index, coordinate in enumerate(CANONICAL_KEYPOINTS)
    }
    maps = []
    for flip_width, flip_length in ((False, False), (True, False), (False, True), (True, True)):
        permutation = []
        for x, y in CANONICAL_KEYPOINTS:
            transformed = (9.0 - x if flip_width else x, 18.0 - y if flip_length else y)
            permutation.append(coordinate_to_index[transformed])
        maps.append(tuple(permutation))
    return tuple(maps)


POSE36_SYMMETRY_MAPS = _pose36_symmetry_maps()


def canonicalize_pose36_points(
    points: Sequence[PointSample], symmetry_index: int
) -> list[PointSample]:
    """Remap a legal Pose36 symmetry to the fixed image-facing identity convention."""

    if not 0 <= symmetry_index < len(POSE36_SYMMETRY_MAPS):
        return list(points)
    permutation = POSE36_SYMMETRY_MAPS[symmetry_index]
    return [
        PointSample(permutation[point.index], point.x, point.y, point.visibility)
        if 0 <= point.index < len(permutation)
        else point
        for point in points
    ]


@dataclass(frozen=True)
class ImageTransform:
    matrix: np.ndarray
    inverse: np.ndarray
    size: int

    @property
    def mirrored(self) -> bool:
        return float(np.linalg.det(self.matrix[:2, :2])) < 0.0

    def apply_segment(self, segment: Sequence[float]) -> tuple[float, float, float, float] | None:
        points = np.asarray(
            [[segment[0], segment[1], 1.0], [segment[2], segment[3], 1.0]],
            dtype=np.float64,
        )
        transformed = (self.matrix @ points.T).T[:, :2]
        return clip_segment_to_image(transformed.reshape(-1).tolist(), self.size, self.size)

    def restore_segment(
        self,
        segment: Sequence[float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        points = np.asarray(
            [[segment[0], segment[1], 1.0], [segment[2], segment[3], 1.0]],
            dtype=np.float64,
        )
        restored = (self.inverse @ points.T).T[:, :2]
        return clip_segment_to_image(restored.reshape(-1).tolist(), width, height)

    def restore_point(self, point: Sequence[float]) -> tuple[float, float]:
        value = self.inverse @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
        return float(value[0]), float(value[1])

    def apply_point(self, point: Sequence[float]) -> tuple[float, float]:
        value = self.matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
        return float(value[0]), float(value[1])


def warp_to_square(
    image: np.ndarray,
    size: int,
    *,
    training: bool = False,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, ImageTransform]:
    if size < 64 or size % 4:
        raise ValueError("image size must be >=64 and divisible by stride 4")
    rng = rng or random.Random()
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    if not training:
        resized_width = max(1, min(size, int(round(width * scale))))
        resized_height = max(1, min(size, int(round(height * scale))))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        warped = np.full((size, size, image.shape[2]), 114, dtype=image.dtype)
        warped[top : top + resized_height, left : left + resized_width] = resized
        matrix = np.asarray(
            [
                [resized_width / width, 0.0, float(left)],
                [0.0, resized_height / height, float(top)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return warped, ImageTransform(matrix=matrix, inverse=np.linalg.inv(matrix), size=size)
    if training:
        scale *= rng.uniform(0.90, 1.10)
    translate_x = 0.5 * (size - width * scale)
    translate_y = 0.5 * (size - height * scale)
    if training:
        translate_x += rng.uniform(-0.06, 0.06) * size
        translate_y += rng.uniform(-0.06, 0.06) * size
    matrix = np.asarray(
        [[scale, 0.0, translate_x], [0.0, scale, translate_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if training and rng.random() < 0.5:
        flip = np.asarray([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        matrix = flip @ matrix
    warped = cv2.warpAffine(
        image,
        matrix[:2],
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    transform = ImageTransform(matrix=matrix, inverse=np.linalg.inv(matrix), size=size)
    return warped, transform


def points_to_layout_target(
    points: Sequence[PointSample],
    transform: ImageTransform,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    """Fit a supervised homography and return its four ordered outer-court corners.

    Frames with insufficient two-dimensional support are deliberately labelled
    unobservable. They train the validity head to abstain instead of inventing a
    complete court from one isolated line or corner.
    """

    canonical = []
    image = []
    for point in points:
        if (
            point.visibility not in {1, 2}
            or not 0.0 <= point.x <= 1.0
            or not 0.0 <= point.y <= 1.0
            or not 0 <= point.index < len(CANONICAL_KEYPOINTS)
        ):
            continue
        court_x, court_y = CANONICAL_KEYPOINTS[point.index]
        if transform.mirrored:
            court_x = 9.0 - court_x
        pixel_x, pixel_y = transform.apply_point((point.x * width, point.y * height))
        canonical.append((court_x / 9.0, court_y / 18.0))
        image.append((pixel_x / transform.size, pixel_y / transform.size))

    invalid = {
        "layout_corners": torch.zeros((4, 2), dtype=torch.float32),
        "layout_valid": torch.tensor(0.0, dtype=torch.float32),
        "layout_fit_error": torch.tensor(0.0, dtype=torch.float32),
        "layout_symmetry": torch.tensor(-1, dtype=torch.int64),
        "layout_points": torch.zeros((len(CANONICAL_KEYPOINTS), 2), dtype=torch.float32),
        "layout_point_valid": torch.zeros(len(CANONICAL_KEYPOINTS), dtype=torch.float32),
    }
    if len(canonical) < 4:
        return invalid
    source = np.asarray(canonical, dtype=np.float64)
    destination = np.asarray(image, dtype=np.float64)
    source_hull = cv2.convexHull(source.astype(np.float32))
    destination_hull = cv2.convexHull(destination.astype(np.float32))
    if abs(float(cv2.contourArea(source_hull))) < 0.01:
        return invalid
    if abs(float(cv2.contourArea(destination_hull))) < 0.002:
        return invalid
    homography, _mask = cv2.findHomography(source, destination, method=0)
    if homography is None or not np.isfinite(homography).all():
        return invalid
    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    projected = (homography @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-8):
        return invalid
    projected = projected[:, :2] / projected[:, 2:3]
    fit_error = float(np.median(np.linalg.norm(projected - destination, axis=1)))
    if not math.isfinite(fit_error) or fit_error > 0.02:
        return invalid

    corner_source = np.asarray(
        [
            (
                CANONICAL_KEYPOINTS[index][0] / 9.0,
                CANONICAL_KEYPOINTS[index][1] / 18.0,
                1.0,
            )
            for index in LAYOUT_CORNER_IDS
        ],
        dtype=np.float64,
    )
    corners = (homography @ corner_source.T).T
    if np.any(np.abs(corners[:, 2]) < 1e-8):
        return invalid
    corners = corners[:, :2] / corners[:, 2:3]
    if not np.isfinite(corners).all() or float(np.max(np.abs(corners))) > 3.0:
        return invalid
    if abs(float(cv2.contourArea(corners.astype(np.float32)))) < 0.005:
        return invalid
    # Pose36 labels may use any of the four legal width/length symmetry
    # conventions.  They describe the same visible court and must not become
    # negative samples for the observability head.  Canonicalize only through
    # those topology-preserving permutations, choosing the convention whose
    # near baseline is lower in the image and whose left side stays left.
    ordered_candidates: list[tuple[float, int, np.ndarray]] = []
    for symmetry_index, permutation in enumerate(LAYOUT_SYMMETRY_PERMUTATIONS):
        candidate = corners[list(permutation)]
        near_y = float(np.mean(candidate[[0, 3], 1]))
        far_y = float(np.mean(candidate[[1, 2], 1]))
        left_x = float(np.mean(candidate[[0, 1], 0]))
        right_x = float(np.mean(candidate[[2, 3], 0]))
        if near_y <= far_y or left_x >= right_x:
            continue
        ordered_candidates.append(
            ((near_y - far_y) + (right_x - left_x), symmetry_index, candidate)
        )
    if not ordered_candidates:
        return invalid
    _, symmetry_index, corners = max(ordered_candidates, key=lambda row: row[0])
    fixed_canonical_corners = np.asarray(
        [[0.0, 0.0], [0.0, 18.0], [9.0, 18.0], [9.0, 0.0]], dtype=np.float32
    )
    fixed_homography = cv2.getPerspectiveTransform(
        fixed_canonical_corners, corners.astype(np.float32)
    )
    all_canonical = np.column_stack(
        (
            np.asarray(CANONICAL_KEYPOINTS, dtype=np.float64),
            np.ones(len(CANONICAL_KEYPOINTS), dtype=np.float64),
        )
    )
    layout_points = (fixed_homography @ all_canonical.T).T
    if np.any(np.abs(layout_points[:, 2]) < 1e-8):
        return invalid
    layout_points = layout_points[:, :2] / layout_points[:, 2:3]
    point_valid = np.logical_and.reduce(
        (
            np.isfinite(layout_points).all(axis=1),
            layout_points[:, 0] >= 0.0,
            layout_points[:, 0] < 1.0,
            layout_points[:, 1] >= 0.0,
            layout_points[:, 1] < 1.0,
        )
    )
    return {
        "layout_corners": torch.from_numpy(corners.astype(np.float32)),
        "layout_valid": torch.tensor(1.0, dtype=torch.float32),
        "layout_fit_error": torch.tensor(fit_error, dtype=torch.float32),
        "layout_symmetry": torch.tensor(symmetry_index, dtype=torch.int64),
        "layout_points": torch.from_numpy(layout_points.astype(np.float32)),
        "layout_point_valid": torch.from_numpy(point_valid.astype(np.float32)),
    }


def _photometric(image: np.ndarray, rng: random.Random) -> np.ndarray:
    output = image
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-5.0, 5.0), 180.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.80, 1.20), 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.75, 1.25), 0.0, 255.0)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.12:
        output = cv2.GaussianBlur(output, (3, 3), rng.uniform(0.2, 1.0))
    if rng.random() < 0.08:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, 4.0, output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    if rng.random() < 0.10:
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(65, 92)])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                output = decoded
    return output


def sample_segment_points(
    segment: Sequence[float],
    spacing: float,
) -> list[tuple[float, float]]:
    """Return midpoint-phased, approximately uniform samples without endpoint identities."""

    if spacing <= 0.0:
        raise ValueError("sample spacing must be positive")
    x1, y1, x2, y2 = map(float, segment)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1.0:
        return []
    count = max(2, int(math.ceil(length / spacing)))
    return [
        (x1 + ((index + 0.5) / count) * dx, y1 + ((index + 0.5) / count) * dy)
        for index in range(count)
    ]


def _finite_intersection(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float] | None:
    p = np.asarray(first[:2], dtype=np.float64)
    r = np.asarray(first[2:], dtype=np.float64) - p
    q = np.asarray(second[:2], dtype=np.float64)
    s = np.asarray(second[2:], dtype=np.float64) - q
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < 1e-6:
        return None
    delta = q - p
    t = float((delta[0] * s[1] - delta[1] * s[0]) / cross)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / cross)
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        point = p + t * r
        return float(point[0]), float(point[1])
    return None


def segments_to_targets(
    segments: Sequence[Sequence[float]],
    image_size: int,
    stride: int = 4,
    target_mode: str = "center",
    sample_spacing: float = 16.0,
    intersection_exclusion: float = 4.0,
    segment_families: Sequence[int] | None = None,
    segment_identities: Sequence[int] | None = None,
    court_roi: np.ndarray | None = None,
    roi_valid: bool = False,
) -> dict[str, torch.Tensor]:
    if target_mode not in {"center", "dense_votes", "dense_context", "dense_semantic"}:
        raise ValueError(f"unsupported target mode: {target_mode}")
    if segment_families is not None and len(segment_families) != len(segments):
        raise ValueError("segment family count must match segment count")
    if segment_identities is not None and len(segment_identities) != len(segments):
        raise ValueError("segment identity count must match segment count")
    grid = image_size // stride
    heatmap = torch.zeros((1, grid, grid), dtype=torch.float32)
    offset = torch.zeros((2, grid, grid), dtype=torch.float32)
    orientation = torch.zeros((2, grid, grid), dtype=torch.float32)
    half_length = torch.zeros((1, grid, grid), dtype=torch.float32)
    family = torch.zeros((grid, grid), dtype=torch.long)
    identity = torch.zeros((grid, grid), dtype=torch.long)
    regression_mask = torch.zeros((1, grid, grid), dtype=torch.float32)
    retained_length: dict[tuple[int, int], float] = {}
    collision_count = 0
    diagonal = math.hypot(image_size, image_size)
    intersections = [
        intersection
        for first_index, first in enumerate(segments)
        for second in segments[first_index + 1 :]
        if (intersection := _finite_intersection(first, second)) is not None
    ]
    for segment_index, segment in enumerate(segments):
        x1, y1, x2, y2 = map(float, segment)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        votes = (
            [(0.5 * (x1 + x2), 0.5 * (y1 + y2))]
            if target_mode == "center"
            else [
                point
                for point in sample_segment_points(segment, sample_spacing)
                if not any(
                    math.hypot(point[0] - crossing[0], point[1] - crossing[1])
                    <= intersection_exclusion
                    for crossing in intersections
                )
            ]
        )
        theta = math.atan2(dy, dx)
        for center_x, center_y in votes:
            grid_x, grid_y = center_x / stride, center_y / stride
            cell_x, cell_y = int(grid_x), int(grid_y)
            if not (0 <= cell_x < grid and 0 <= cell_y < grid):
                continue
            key = (cell_y, cell_x)
            if key in retained_length:
                collision_count += 1
                if length <= retained_length[key]:
                    continue
            retained_length[key] = length
            heatmap[0, cell_y, cell_x] = 1.0
            offset[:, cell_y, cell_x] = torch.tensor((grid_x - cell_x, grid_y - cell_y))
            orientation[:, cell_y, cell_x] = torch.tensor(
                (math.cos(2.0 * theta), math.sin(2.0 * theta))
            )
            family[cell_y, cell_x] = int(segment_families[segment_index]) if segment_families else 0
            identity[cell_y, cell_x] = (
                int(segment_identities[segment_index]) if segment_identities else 0
            )
            half_length[0, cell_y, cell_x] = 0.5 * length / diagonal
            regression_mask[0, cell_y, cell_x] = 1.0
    if court_roi is None:
        roi_target = torch.zeros((1, grid, grid), dtype=torch.float32)
    else:
        resized_roi = cv2.resize(
            court_roi.astype(np.float32),
            (grid, grid),
            interpolation=cv2.INTER_AREA,
        )
        roi_target = torch.from_numpy(np.clip(resized_roi, 0.0, 1.0)).unsqueeze(0)
    return {
        "heatmap": heatmap,
        "offset": offset,
        "orientation": orientation,
        "half_length": half_length,
        "regression_mask": regression_mask,
        "family": family,
        "identity": identity,
        "court_roi": roi_target,
        "roi_valid": torch.tensor(float(roi_valid), dtype=torch.float32),
        "collision_count": torch.tensor(collision_count, dtype=torch.int64),
        "vote_count": regression_mask.sum().to(torch.int64),
    }


class CourtLineDataset(Dataset):
    def __init__(
        self,
        converted_root: str | Path,
        split: str,
        image_size: int = 640,
        image_root: str | Path | None = None,
        augment: bool = False,
        seed: int = 0,
        limit: int = 0,
        target_mode: str = "center",
        sample_spacing: float = 16.0,
        intersection_exclusion: float = 4.0,
        hard_negative_probability: float = 0.0,
    ) -> None:
        self.root = Path(converted_root).resolve()
        self.split = split
        self.image_size = image_size
        self.image_root = Path(image_root).resolve() if image_root else None
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.target_mode = target_mode
        self.sample_spacing = sample_spacing
        self.intersection_exclusion = intersection_exclusion
        if not 0.0 <= hard_negative_probability <= 1.0:
            raise ValueError("hard-negative probability must be between zero and one")
        self.hard_negative_probability = hard_negative_probability
        self.labels = sorted((self.root / split / "labels").glob("*.json"))
        if limit > 0:
            self.labels = self.labels[:limit]
        if not self.labels:
            raise FileNotFoundError(f"no converted labels under {self.root / split / 'labels'}")

    def __len__(self) -> int:
        return len(self.labels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _image_path(self, metadata: dict[str, Any]) -> Path:
        absolute = Path(metadata["image"])
        if absolute.is_file():
            return absolute
        if self.image_root and metadata.get("image_relative"):
            candidate = self.image_root / metadata["image_relative"]
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"image is unavailable: {absolute}; pass --image-root for portable converted labels"
        )

    def _points(self, metadata: dict[str, Any]) -> list[PointSample]:
        embedded = metadata.get("keypoints")
        if embedded:
            return [
                PointSample(
                    int(row["id"]),
                    float(row["x"]),
                    float(row["y"]),
                    int(row["visibility"]),
                )
                for row in embedded
            ]
        candidates = [Path(str(metadata.get("pose_label", "")))]
        if self.image_root and metadata.get("image_relative"):
            relative = Path(str(metadata["image_relative"]))
            parts = list(relative.parts)
            if "images" in parts:
                parts[parts.index("images")] = "labels"
                candidates.append((self.image_root / Path(*parts)).with_suffix(".txt"))
        for candidate in candidates:
            if candidate.is_file():
                rows = [
                    row for row in candidate.read_text(encoding="utf-8").splitlines() if row.strip()
                ]
                return parse_yolo_pose_line(rows[0]) if rows else []
        return []

    def _court_roi(
        self,
        points: Sequence[PointSample],
        transform: ImageTransform,
        width: int,
        height: int,
    ) -> tuple[np.ndarray | None, bool]:
        valid = {
            point.index: transform.apply_point((point.x * width, point.y * height))
            for point in points
            if point.visibility in {1, 2} and 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0
        }
        corners = (0, 9, 5, 4)
        if all(index in valid for index in corners):
            polygon = np.asarray([valid[index] for index in corners], dtype=np.float32)
        elif len(valid) >= 3:
            polygon = cv2.convexHull(np.asarray(list(valid.values()), dtype=np.float32)).reshape(
                -1, 2
            )
        else:
            return None, False
        if abs(float(cv2.contourArea(polygon.reshape(-1, 1, 2)))) < 32.0:
            return None, False
        roi = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 1.0)
        return roi, True

    def _hard_negative(
        self,
        image: np.ndarray,
        roi: np.ndarray | None,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if roi is None:
            return None
        size = self.image_size
        for _ in range(20):
            crop_width = rng.randint(max(64, size // 3), max(65, int(size * 0.7)))
            crop_height = rng.randint(max(64, size // 3), max(65, int(size * 0.7)))
            left = rng.randint(0, size - crop_width)
            top = rng.randint(0, size - crop_height)
            crop_roi = roi[top : top + crop_height, left : left + crop_width]
            if float(crop_roi.mean()) > 0.015:
                continue
            crop = image[top : top + crop_height, left : left + crop_width]
            negative = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.7:
                for _line in range(rng.randint(1, 5)):
                    first = (rng.randrange(size), rng.randrange(size))
                    second = (rng.randrange(size), rng.randrange(size))
                    color = tuple(rng.randint(120, 255) for _channel in range(3))
                    cv2.line(negative, first, second, color, rng.randint(1, 4), cv2.LINE_AA)
            return negative, np.zeros((size, size), dtype=np.float32)
        return None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metadata = json.loads(self.labels[index].read_text(encoding="utf-8"))
        image_path = self._image_path(metadata)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode: {image_path}")
        rng = random.Random((self.seed + 1) * 1_000_003 + self.epoch * len(self) + index)
        warped, transform = warp_to_square(
            image,
            self.image_size,
            training=self.augment,
            rng=rng,
        )
        segments = []
        families = []
        identities = []
        for row in metadata.get("line_groups", []):
            if not row.get("valid") or row.get("segment") is None:
                continue
            transformed = transform.apply_segment(row["segment"])
            if transformed is not None:
                segments.append(transformed)
                families.append(0 if int(row.get("topology_index", -1)) in {0, 2} else 1)
                identity = int(row.get("topology_index", 0))
                if transform.mirrored:
                    identity = {0: 2, 2: 0}.get(identity, identity)
                identities.append(identity)
        points = self._points(metadata)
        court_roi, roi_valid = self._court_roi(points, transform, image.shape[1], image.shape[0])
        layout_target = points_to_layout_target(
            points,
            transform,
            image.shape[1],
            image.shape[0],
        )
        is_hard_negative = False
        if (
            self.augment
            and self.target_mode in {"dense_context", "dense_semantic"}
            and rng.random() < self.hard_negative_probability
        ):
            negative = self._hard_negative(warped, court_roi, rng)
            if negative is not None:
                warped, court_roi = negative
                roi_valid = True
                segments = []
                families = []
                identities = []
                is_hard_negative = True
                layout_target["layout_valid"] = torch.tensor(0.0, dtype=torch.float32)
                layout_target["layout_point_valid"].zero_()
        if self.augment:
            warped = _photometric(warped, rng)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        )
        return image_tensor, segments_to_targets(
            segments,
            self.image_size,
            target_mode=self.target_mode,
            sample_spacing=self.sample_spacing,
            intersection_exclusion=self.intersection_exclusion,
            segment_families=families,
            segment_identities=identities,
            court_roi=court_roi,
            roi_valid=roi_valid,
        ) | {
            "hard_negative": torch.tensor(float(is_hard_negative), dtype=torch.float32),
        } | layout_target
