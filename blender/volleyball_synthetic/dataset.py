"""Dataset rendering loop, metadata manifest, and .blend project export."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import bpy

from .capture import (
    _apply_capture_profile,
    _mark_broadcast_ui_occlusion,
    _sample_capture_profile,
    _save_degraded_render,
    configure_render,
)
from .config import load_randomization_config
from .constants import KEYPOINT_NAMES, KEYPOINT_WORLD
from .labels import (
    _apply_keypoint_permutation,
    _camera_intrinsics,
    _keypoint_labels,
    _matrix_rows,
    _split_name,
    _write_dataset_yaml,
    _yolo_pose_line,
)
from .randomization import randomize_scene
from .scene_builder import build_scene

def generate_dataset(
    state: dict[str, Any],
    output: Path,
    *,
    count: int,
    seed: int,
    overwrite: bool = False,
    minimum_resolution_y: int = 480,
    randomize_capture_quality: bool = True,
) -> dict[str, Any]:
    output = output.resolve()
    existing = list(output.glob("*/images/*")) if output.exists() else []
    if existing and not overwrite:
        raise FileExistsError(f"dataset already contains {len(existing)} images: {output}; pass --overwrite explicitly")
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels", "metadata"):
            (output / split / kind).mkdir(parents=True, exist_ok=True)
    _write_dataset_yaml(output)
    config_snapshot_path = output / "randomization-config.yaml"
    config_snapshot_path.write_text(
        Path(state["config"]["_config_path"]).read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )

    scene = state["scene"]
    camera = state["camera"]
    base_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    base_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    split_counts: Counter[str] = Counter()
    visibility_counts: Counter[int] = Counter()
    resolution_counts: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    blur_counts: Counter[str] = Counter()
    arena_profile_counts: Counter[str] = Counter()
    player_pattern_counts: Counter[str] = Counter()
    source_camera_position_counts: Counter[str] = Counter()
    canonical_camera_mode_counts: Counter[str] = Counter()
    fake_ui_count = 0
    rows: list[dict[str, Any]] = []
    for index in range(count):
        # Specialized generators may use the stable sample index to schedule
        # assets exactly while every other randomizer continues using the seed.
        state["sample_index"] = index
        sample_seed = (seed * 1_000_003 + index * 97_409) & 0x7FFFFFFF
        capture_rng = random.Random(sample_seed ^ 0xC4A7E123)
        capture = _sample_capture_profile(
            capture_rng,
            config=state["config"],
            base_width=base_width,
            base_height=base_height,
            minimum_height=minimum_resolution_y,
            enabled=randomize_capture_quality,
        )
        _apply_capture_profile(scene, capture)
        scene_info = randomize_scene(state, sample_seed)
        scene.view_settings.exposure += float(capture["exposure_offset"])
        width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
        height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
        raw_keypoints = _keypoint_labels(scene, camera)
        keypoint_permutation = [int(value) for value in state["keypoint_permutation"]]
        keypoints = _apply_keypoint_permutation(raw_keypoints, keypoint_permutation)
        capture["ui_occluded_keypoints"] = _mark_broadcast_ui_occlusion(keypoints, width, height, capture)
        split = _split_name(sample_seed, state["config"])
        stem = f"synthetic_{index:06d}_{sample_seed:010d}"
        image_path = output / split / "images" / f"{stem}.jpg"
        label_path = output / split / "labels" / f"{stem}.txt"
        metadata_path = output / split / "metadata" / f"{stem}.json"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        _save_degraded_render(image_path, capture, sample_seed)
        label_path.write_text(_yolo_pose_line(keypoints, width, height) + "\n", encoding="utf-8")
        camera_to_world = camera.matrix_world.copy()
        metadata = {
            "schema": "court36-synthetic-blender-v2",
            "sample_index": index,
            "sample_seed": sample_seed,
            "split": split,
            "image": image_path.relative_to(output).as_posix(),
            "label": label_path.relative_to(output).as_posix(),
            "scene": scene_info,
            "capture": capture,
            "source_camera_position": state["source_camera_position"],
            "canonical_camera_mode": state["canonical_camera_mode"],
            "keypoint_permutation": keypoint_permutation,
            "camera": {
                "intrinsics": _camera_intrinsics(scene, camera),
                "camera_to_world": _matrix_rows(camera_to_world),
                "world_to_camera": _matrix_rows(camera_to_world.inverted()),
                "location": [float(value) for value in camera.location],
                "rotation_quaternion": [float(value) for value in camera.rotation_quaternion],
            },
            "keypoints": keypoints,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        split_counts[split] += 1
        visibility_counts.update(int(row["visibility"]) for row in keypoints)
        resolution_counts[f"{width}x{height}"] += 1
        filter_counts[str(capture["filter"])] += 1
        blur_counts[str(capture["blur_mode"])] += 1
        arena_profile_counts[str(scene_info["arena_profile"])] += 1
        player_pattern_counts[str(scene_info["player_pattern"])] += 1
        source_camera_position_counts[str(state["source_camera_position"])] += 1
        canonical_camera_mode_counts[str(state["canonical_camera_mode"])] += 1
        fake_ui_count += int(capture["fake_ui_enabled"])
        rows.append(
            {
                "index": index,
                "seed": sample_seed,
                "split": split,
                "image": image_path.relative_to(output).as_posix(),
                "visible": sum(int(row["visibility"]) == 2 for row in keypoints),
                "occluded": sum(int(row["visibility"]) == 1 for row in keypoints),
                "outside": sum(int(row["visibility"]) == 0 for row in keypoints),
                "resolution": [width, height],
                "capture": capture,
                **scene_info,
            }
        )
        print(
            f"[{index + 1}/{count}] {split}/{stem} {width}x{height} q={capture['jpeg_quality']} "
            f"filter={capture['filter']} blur={capture['blur_mode']} ui={int(capture['fake_ui_enabled'])} "
            f"v2={rows[-1]['visible']} v1={rows[-1]['occluded']} v0={rows[-1]['outside']}",
            flush=True,
        )

    scene.render.resolution_x = base_width
    scene.render.resolution_y = base_height
    scene.render.image_settings.quality = 94
    scene.view_settings.use_white_balance = False
    scene.view_settings.white_balance_temperature = 6500.0
    scene.view_settings.white_balance_tint = 10.0
    scene.view_settings.gamma = 1.0

    manifest = {
        "schema": "court36-synthetic-blender-v2",
        "generator": str(Path(__file__).resolve()),
        "blender_version": bpy.app.version_string,
        "count": count,
        "seed": seed,
        "resolution": {
            "mode": "variable" if randomize_capture_quality else "fixed",
            "maximum": [base_width, base_height],
            "minimum_height": min(base_height, minimum_resolution_y),
            "counts": dict(sorted(resolution_counts.items())),
        },
        "capture_randomization": randomize_capture_quality,
        "randomization_config": state["config"]["_config_path"],
        "randomization_config_snapshot": config_snapshot_path.name,
        "selection_weight_mode": state["config"]["selection_weight_mode"],
        "yaml_parser": state["config"]["_yaml_parser"],
        "filter_counts": dict(sorted(filter_counts.items())),
        "blur_counts": dict(sorted(blur_counts.items())),
        "arena_profile_counts": dict(sorted(arena_profile_counts.items())),
        "player_pattern_counts": dict(sorted(player_pattern_counts.items())),
        "source_camera_position_counts": dict(sorted(source_camera_position_counts.items())),
        "canonical_camera_mode_counts": dict(sorted(canonical_camera_mode_counts.items())),
        "fake_ui_count": fake_ui_count,
        "fictional_sponsor_atlas": "blender/assets/fictional-sponsor-led-atlas-v1.png",
        "split_counts": dict(split_counts),
        "visibility_counts": {str(key): value for key, value in sorted(visibility_counts.items())},
        "keypoint_names": list(KEYPOINT_NAMES),
        "coordinate_convention": "x=width-left-to-right,y=length-far-to-near,z=up-metres",
        "samples": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

def create_and_save_project(
    blend_path: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int = 20260811,
    width: int = 1280,
    height: int = 720,
    samples: int = 48,
    engine: str = "BLENDER_EEVEE",
) -> dict[str, Any]:
    """MCP-friendly entry point that builds a ready-to-render .blend file."""

    config = load_randomization_config(config_path)
    state = build_scene(seed, config)
    configure_render(state["scene"], config=config, width=width, height=height, samples=samples, engine=engine)
    preview = randomize_scene(state, seed)
    destination = Path(blend_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    return {
        "blend_file": str(destination),
        "objects": len(bpy.context.scene.objects),
        "keypoints": len(KEYPOINT_WORLD),
        "net_height": state["net_height"],
        "randomization_config": config["_config_path"],
        "yaml_parser": config["_yaml_parser"],
        "preview": preview,
    }
