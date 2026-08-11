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


@dataclass(frozen=True)
class ImageTransform:
    matrix: np.ndarray
    inverse: np.ndarray
    size: int

    def apply_segment(self, segment: Sequence[float]) -> tuple[float, float, float, float] | None:
        points = np.asarray(
            [[segment[0], segment[1], 1.0], [segment[2], segment[3], 1.0]],
            dtype=np.float64,
        )
        transformed = (self.matrix @ points.T).T[:, :2]
        return clip_segment_to_image(transformed.reshape(-1), self.size, self.size)

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
        return clip_segment_to_image(restored.reshape(-1), width, height)

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
                identities.append(int(row.get("topology_index", 0)))
        points = self._points(metadata)
        court_roi, roi_valid = self._court_roi(points, transform, image.shape[1], image.shape[0])
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
        }
