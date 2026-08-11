from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LineGroup:
    """One physical line, used only while converting point annotations."""

    point_indices: tuple[int, ...]
    endpoint_indices: tuple[int, int]


@dataclass(frozen=True)
class CourtLineTopology:
    version: int
    keypoint_count: int
    usable_visibility: frozenset[int]
    line_groups: tuple[LineGroup, ...]


def _line_group(row: dict[str, Any], keypoint_count: int) -> LineGroup:
    points = tuple(int(value) for value in row.get("points", ()))
    endpoints = tuple(int(value) for value in row.get("endpoint_indices", ()))
    if len(points) < 2:
        raise ValueError("each line group must contain at least two point indices")
    if len(set(points)) != len(points):
        raise ValueError(f"line group contains duplicate point indices: {points}")
    if len(endpoints) != 2:
        raise ValueError(f"line group must contain exactly two endpoint indices: {endpoints}")
    if any(index < 0 or index >= keypoint_count for index in (*points, *endpoints)):
        raise ValueError(f"line group index outside [0, {keypoint_count}): {row}")
    if any(index not in points for index in endpoints):
        raise ValueError(f"endpoint indices must be members of the line group: {row}")
    return LineGroup(points, (endpoints[0], endpoints[1]))


def load_topology(path: str | Path) -> CourtLineTopology:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"topology must be a YAML mapping: {source}")
    version = int(payload.get("version", 0))
    keypoint_count = int(payload.get("keypoint_count", 0))
    if version < 1:
        raise ValueError(f"unsupported topology version: {version}")
    if keypoint_count < 2:
        raise ValueError(f"invalid keypoint_count: {keypoint_count}")
    visibility = frozenset(int(value) for value in payload.get("usable_visibility", ()))
    if not visibility or any(value <= 0 for value in visibility):
        raise ValueError(
            f"usable_visibility must contain positive YOLO visibility values: {visibility}"
        )
    raw_groups = payload.get("line_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("topology must contain a non-empty line_groups list")
    groups = tuple(_line_group(row, keypoint_count) for row in raw_groups)
    referenced = {index for group in groups for index in group.point_indices}
    missing = sorted(set(range(keypoint_count)) - referenced)
    if missing:
        raise ValueError(f"topology does not reference keypoints: {missing}")
    return CourtLineTopology(version, keypoint_count, visibility, groups)
