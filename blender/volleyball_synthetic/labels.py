"""Camera projection, occlusion, and canonical YOLO Pose label export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from .config import _entry_weight
from .constants import COURT_WIDTH, KEYPOINT_NAMES, KEYPOINT_WORLD


def _apply_keypoint_permutation(
    keypoints: list[dict[str, Any]],
    permutation: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Move each complete keypoint row into its canonical Court36 index."""

    if len(keypoints) != 36 or len(permutation) != 36:
        raise ValueError(
            f"Court36 permutation requires 36 rows and 36 indices, got {len(keypoints)} and {len(permutation)}"
        )
    if sorted(int(index) for index in permutation) != list(range(36)):
        raise ValueError("Court36 keypoint permutation must contain each source index exactly once")
    output: list[dict[str, Any]] = []
    for output_index, source_index_value in enumerate(permutation):
        source_index = int(source_index_value)
        source = keypoints[source_index]
        row = dict(source)
        row["index"] = output_index
        row["name"] = KEYPOINT_NAMES[output_index]
        row["source_index"] = source_index
        row["source_name"] = source["name"]
        output.append(row)
    return output

def _project_point(scene: bpy.types.Scene, camera: bpy.types.Object, point: Vector) -> tuple[float, float, float]:
    normalized = world_to_camera_view(scene, camera, point)
    width = float(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    height = float(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    return normalized.x * width, (1.0 - normalized.y) * height, normalized.z

def _matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]

def _camera_intrinsics(scene: bpy.types.Scene, camera: bpy.types.Object) -> dict[str, Any]:
    width = float(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    height = float(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    pixel_aspect = float(scene.render.pixel_aspect_y / scene.render.pixel_aspect_x)
    sensor_fit = camera.data.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= height * pixel_aspect else "VERTICAL"
    sensor_size = camera.data.sensor_height if sensor_fit == "VERTICAL" else camera.data.sensor_width
    view_fac = pixel_aspect * height if sensor_fit == "VERTICAL" else width
    focal_px = camera.data.lens / sensor_size * view_fac
    fx = focal_px
    fy = focal_px / pixel_aspect
    cx = width * 0.5 - camera.data.shift_x * view_fac
    cy = height * 0.5 + camera.data.shift_y * view_fac / pixel_aspect
    return {
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(camera.data.sensor_width),
        "sensor_height_mm": float(camera.data.sensor_height),
        "sensor_fit": sensor_fit,
        "shift_x": float(camera.data.shift_x),
        "shift_y": float(camera.data.shift_y),
        "resolution": [int(width), int(height)],
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }

def _ray_occluder(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    origin: Vector,
    target: Vector,
) -> str | None:
    current = origin.copy()
    for _ in range(8):
        ray = target - current
        if ray.length <= 0.055:
            return None
        direction = ray.normalized()
        hit, location, _normal, _face_index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            current,
            direction,
            distance=max(0.0, ray.length - 0.055),
        )
        if not hit:
            return None
        if hit_object is not None and hit_object.get("synthetic_ray_ignore"):
            current = location + direction * 0.02
            continue
        return hit_object.name if hit_object is not None else "unknown"
    return None

def _net_texture_occludes(scene: bpy.types.Scene, origin: Vector, target: Vector) -> bool:
    delta_y = target.y - origin.y
    if abs(delta_y) < 1e-8:
        return False
    t = (9.0 - origin.y) / delta_y
    if not 0.0 < t < 1.0:
        return False
    intersection = origin + (target - origin) * t
    net_height = float(scene.get("net_height", 2.43))
    bottom = net_height - 1.0
    if not (0.0 <= intersection.x <= COURT_WIDTH and bottom <= intersection.z <= net_height):
        return False
    x_spacing = COURT_WIDTH / 90.0
    z_spacing = 1.0 / 10.0
    x_distance = abs(intersection.x / x_spacing - round(intersection.x / x_spacing)) * x_spacing
    z_distance = abs((intersection.z - bottom) / z_spacing - round((intersection.z - bottom) / z_spacing)) * z_spacing
    return x_distance <= 0.010 or z_distance <= 0.010

def _keypoint_labels(scene: bpy.types.Scene, camera: bpy.types.Object) -> list[dict[str, Any]]:
    width = float(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    height = float(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation.copy()
    rows: list[dict[str, Any]] = []
    for index, (name, raw_point) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_WORLD)):
        point = Vector(raw_point)
        pixel_x, pixel_y, depth = _project_point(scene, camera, point)
        in_frame = depth > 0.0 and 0.0 <= pixel_x < width and 0.0 <= pixel_y < height
        visibility = 0
        occluded_by: str | None = None
        if in_frame:
            occluded_by = _ray_occluder(scene, depsgraph, origin, point)
            if occluded_by is not None:
                visibility = 1
            elif _net_texture_occludes(scene, origin, point):
                visibility = 1
                occluded_by = "NET_texture_grid"
            else:
                visibility = 2
        rows.append(
            {
                "index": index,
                "name": name,
                "world": [float(value) for value in point],
                "pixel": [float(pixel_x), float(pixel_y)] if in_frame else None,
                "visibility": visibility,
                "occluded_by": occluded_by,
            }
        )
    return rows

def _yolo_pose_line(keypoints: list[dict[str, Any]], width: int, height: int) -> str:
    active = [row for row in keypoints if row["visibility"] > 0 and row["pixel"] is not None]
    if len(active) < 2:
        raise RuntimeError("a synthetic frame must contain at least two in-frame court keypoints")
    xs = [float(row["pixel"][0]) for row in active]
    ys = [float(row["pixel"][1]) for row in active]
    pad_x = max(4.0, (max(xs) - min(xs)) * 0.025)
    pad_y = max(4.0, (max(ys) - min(ys)) * 0.025)
    x1 = max(0.0, min(xs) - pad_x)
    y1 = max(0.0, min(ys) - pad_y)
    x2 = min(float(width), max(xs) + pad_x)
    y2 = min(float(height), max(ys) + pad_y)
    values = [
        "0",
        f"{(x1 + x2) * 0.5 / width:.8f}",
        f"{(y1 + y2) * 0.5 / height:.8f}",
        f"{max(1.0, x2 - x1) / width:.8f}",
        f"{max(1.0, y2 - y1) / height:.8f}",
    ]
    for row in keypoints:
        visibility = int(row["visibility"])
        if visibility > 0 and row["pixel"] is not None:
            values.extend(
                (
                    f"{float(row['pixel'][0]) / width:.8f}",
                    f"{float(row['pixel'][1]) / height:.8f}",
                    str(visibility),
                )
            )
        else:
            values.extend(("0", "0", "0"))
    return " ".join(values)

def _split_name(sample_seed: int, config: dict[str, Any]) -> str:
    split_entries = config["splits"]
    names = list(split_entries)
    weights = [_entry_weight(config, split_entries[name], f"splits.{name}") for name in names]
    if any(weight < 0 or not float(weight).is_integer() for weight in weights):
        raise ValueError("split weights must be non-negative integers")
    total = int(sum(weights))
    if total <= 0:
        raise ValueError("at least one split must have a positive weight")
    bucket = sample_seed % total
    cumulative = 0
    for name, weight in zip(names, weights):
        cumulative += int(weight)
        if bucket < cumulative:
            return name
    raise RuntimeError("split selection fell through unexpectedly")

def _write_dataset_yaml(root: Path) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "kpt_shape: [36, 3]",
        "flip_idx: [" + ", ".join(str(index) for index in range(36)) + "]",
        "kpt_names:",
        *(f"  - {name}" for name in KEYPOINT_NAMES),
        "nc: 1",
        "names: [volleyball-court]",
        "",
    ]
    (root / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")
