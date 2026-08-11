"""Per-frame materials, cameras, people, occluders, arena, and lighting randomization."""

from __future__ import annotations

import math
import random
from typing import Any

import bpy
from mathutils import Vector

from .camera_modes import (
    canonical_camera_mode,
    classify_outside_camera_position,
    classify_projected_camera_position,
    keypoint_permutation,
)
from .config import _randint, _range_pair, _uniform, _weighted_entry
from .constants import COURT_CENTER, COURT_LENGTH, COURT_WIDTH, KEYPOINT_WORLD
from .labels import _project_point
from .scene_builder import _look_at, _set_principled_input
from .sky_sphere import randomize_sky_sphere

def _set_noisy_color(material: bpy.types.Material, color: tuple[float, float, float, float], strength: float) -> None:
    material.diffuse_color = color
    ramp = material.node_tree.nodes.get("NoiseRamp") if material.node_tree else None
    if ramp is not None:
        ramp.color_ramp.elements[0].color = tuple(max(0.0, channel * (1.0 - strength)) for channel in color[:3]) + (1.0,)
        ramp.color_ramp.elements[1].color = tuple(min(1.0, channel * (1.0 + strength)) for channel in color[:3]) + (1.0,)
    else:
        _set_principled_input(material, "Base Color", color)

def _randomize_materials(state: dict[str, Any], rng: random.Random) -> str:
    config = state["config"]
    material_config = config["materials"]
    palette_name, palette = _weighted_entry(
        rng,
        config,
        material_config["palettes"],
        "materials.palettes",
    )
    materials = state["materials"]
    _set_noisy_color(
        materials["court"],
        tuple(palette["court"]),
        _uniform(rng, material_config["court_color_noise_strength"], "materials.court_color_noise_strength"),
    )
    _set_noisy_color(
        materials["surround"],
        tuple(palette["surround"]),
        _uniform(rng, material_config["surround_color_noise_strength"], "materials.surround_color_noise_strength"),
    )
    _set_principled_input(materials["team_a"], "Base Color", palette["team_a"])
    _set_principled_input(materials["team_b"], "Base Color", palette["team_b"])
    _set_principled_input(
        materials["court"],
        "Roughness",
        _uniform(rng, material_config["court_roughness"], "materials.court_roughness"),
    )
    _set_principled_input(
        materials["surround"],
        "Roughness",
        _uniform(rng, material_config["surround_roughness"], "materials.surround_roughness"),
    )
    _set_principled_input(
        materials["court"],
        "Coat Weight",
        _uniform(rng, material_config["court_coat_weight"], "materials.court_coat_weight"),
    )
    _set_principled_input(
        materials["surround"],
        "Coat Weight",
        _uniform(rng, material_config["surround_coat_weight"], "materials.surround_coat_weight"),
    )
    _set_principled_input(
        materials["court"],
        "Coat Roughness",
        _uniform(rng, material_config["court_coat_roughness"], "materials.court_coat_roughness"),
    )
    _set_principled_input(
        materials["surround"],
        "Coat Roughness",
        _uniform(rng, material_config["surround_coat_roughness"], "materials.surround_coat_roughness"),
    )
    for material_name in ("court", "surround", "arena"):
        noise = materials[material_name].node_tree.nodes.get("SurfaceNoise")
        if noise is not None:
            noise.inputs["Scale"].default_value = _uniform(
                rng, material_config["surface_noise_scale"], "materials.surface_noise_scale"
            )
            noise.inputs["Detail"].default_value = _uniform(
                rng, material_config["surface_noise_detail"], "materials.surface_noise_detail"
            )
    return palette_name

def _sample_camera_location(
    rng: random.Random,
    net_height: float,
    config: dict[str, Any],
    arena_profile: str = "championship",
) -> tuple[str, Vector, Vector | None]:
    camera_config = config["camera"]
    mode, mode_config = _weighted_entry(rng, config, camera_config["modes"], "camera.modes")
    left_side_probability = float(camera_config["left_side_probability"])
    if mode == "championship_main":
        side = -1.0 if rng.random() < left_side_probability else 1.0
        x = _uniform(rng, mode_config["left_x"], "camera.championship_main.left_x") if side < 0 else _uniform(rng, mode_config["right_x"], "camera.championship_main.right_x")
        return mode, Vector((x, _uniform(rng, mode_config["y"], "camera.championship_main.y"), _uniform(rng, mode_config["z"], "camera.championship_main.z"))), None
    if mode == "broadcast_sideline":
        side = -1.0 if rng.random() < left_side_probability else 1.0
        x = _uniform(rng, mode_config["left_x"], "camera.broadcast_sideline.left_x") if side < 0 else _uniform(rng, mode_config["right_x"], "camera.broadcast_sideline.right_x")
        return mode, Vector((x, _uniform(rng, mode_config["y"], "camera.broadcast_sideline.y"), _uniform(rng, mode_config["z"], "camera.broadcast_sideline.z"))), None
    if mode == "diagonal_sideline":
        side = -1.0 if rng.random() < left_side_probability else 1.0
        x = _uniform(rng, mode_config["left_x"], "camera.diagonal_sideline.left_x") if side < 0 else _uniform(rng, mode_config["right_x"], "camera.diagonal_sideline.right_x")
        return mode, Vector((x, _uniform(rng, mode_config["y"], "camera.diagonal_sideline.y"), _uniform(rng, mode_config["z"], "camera.diagonal_sideline.z"))), None
    if mode == "endline":
        near = rng.random() < float(camera_config["near_end_probability"])
        if arena_profile == "small_gym":
            y_key = "small_gym_near_y" if near else "small_gym_far_y"
        else:
            y_key = "other_near_y" if near else "other_far_y"
        y = _uniform(rng, mode_config[y_key], f"camera.endline.{y_key}")
        return mode, Vector((_uniform(rng, mode_config["x"], "camera.endline.x"), y, _uniform(rng, mode_config["z"], "camera.endline.z"))), None
    if mode == "server_wide":
        far_server = rng.random() < float(camera_config["far_server_probability"])
        focus_y = float(mode_config["far_focus_y"] if far_server else mode_config["near_focus_y"])
        camera_y_key = "far_camera_y" if far_server else "near_camera_y"
        camera_y = _uniform(rng, mode_config[camera_y_key], f"camera.server_wide.{camera_y_key}")
        camera_x = (
            _uniform(rng, mode_config["left_x"], "camera.server_wide.left_x")
            if rng.random() < left_side_probability
            else _uniform(rng, mode_config["right_x"], "camera.server_wide.right_x")
        )
        focus = Vector((_uniform(rng, mode_config["focus_x"], "camera.server_wide.focus_x"), focus_y, _uniform(rng, mode_config["focus_z"], "camera.server_wide.focus_z")))
        return mode, Vector((camera_x, camera_y, _uniform(rng, mode_config["camera_z"], "camera.server_wide.camera_z"))), focus
    if mode == "server_closeup":
        far_server = rng.random() < float(camera_config["far_server_probability"])
        focus_y = float(mode_config["far_focus_y"] if far_server else mode_config["near_focus_y"])
        camera_y_key = "far_camera_y" if far_server else "near_camera_y"
        camera_y = _uniform(rng, mode_config[camera_y_key], f"camera.server_closeup.{camera_y_key}")
        camera_x = (
            _uniform(rng, mode_config["left_x"], "camera.server_closeup.left_x")
            if rng.random() < left_side_probability
            else _uniform(rng, mode_config["right_x"], "camera.server_closeup.right_x")
        )
        focus = Vector((_uniform(rng, mode_config["focus_x"], "camera.server_closeup.focus_x"), focus_y, _uniform(rng, mode_config["focus_z"], "camera.server_closeup.focus_z")))
        return mode, Vector((camera_x, camera_y, _uniform(rng, mode_config["camera_z"], "camera.server_closeup.camera_z"))), focus
    if mode == "net_aligned_server_pan":
        far_server = rng.random() < float(camera_config["far_server_probability"])
        focus_y = float(mode_config["far_focus_y"] if far_server else mode_config["near_focus_y"])
        net_side = -1.0 if rng.random() < left_side_probability else 1.0
        camera_x = _uniform(rng, mode_config["left_x"], "camera.net_aligned_server_pan.left_x") if net_side < 0 else _uniform(rng, mode_config["right_x"], "camera.net_aligned_server_pan.right_x")
        camera_y = _uniform(rng, mode_config["camera_y"], "camera.net_aligned_server_pan.camera_y")
        corner_x = (
            _uniform(rng, mode_config["left_focus_x"], "camera.net_aligned_server_pan.left_focus_x")
            if rng.random() < float(camera_config["server_focus_left_probability"])
            else _uniform(rng, mode_config["right_focus_x"], "camera.net_aligned_server_pan.right_focus_x")
        )
        focus = Vector((corner_x, focus_y, _uniform(rng, mode_config["focus_z"], "camera.net_aligned_server_pan.focus_z")))
        height_low, height_high = _range_pair(mode_config["camera_height_above_net"], "camera.net_aligned_server_pan.camera_height_above_net")
        camera_z = rng.uniform(net_height + height_low, net_height + height_high)
        return mode, Vector((camera_x, camera_y, camera_z)), focus
    if mode == "handheld_sideline":
        side = -1.0 if rng.random() < left_side_probability else 1.0
        x = _uniform(rng, mode_config["left_x"], "camera.handheld_sideline.left_x") if side < 0 else _uniform(rng, mode_config["right_x"], "camera.handheld_sideline.right_x")
        return mode, Vector((x, _uniform(rng, mode_config["y"], "camera.handheld_sideline.y"), _uniform(rng, mode_config["z"], "camera.handheld_sideline.z"))), None
    if mode == "corner":
        x = (
            _uniform(rng, mode_config["left_x"], "camera.corner.left_x")
            if rng.random() < left_side_probability
            else _uniform(rng, mode_config["right_x"], "camera.corner.right_x")
        )
        if arena_profile == "small_gym":
            y = (
                _uniform(rng, mode_config["small_gym_far_y"], "camera.corner.small_gym_far_y")
                if rng.random() < float(camera_config["corner_far_probability"])
                else _uniform(rng, mode_config["small_gym_near_y"], "camera.corner.small_gym_near_y")
            )
        else:
            y = (
                _uniform(rng, mode_config["other_far_y"], "camera.corner.other_far_y")
                if rng.random() < float(camera_config["corner_far_probability"])
                else _uniform(rng, mode_config["other_near_y"], "camera.corner.other_near_y")
            )
        return mode, Vector((x, y, _uniform(rng, mode_config["z"], "camera.corner.z"))), None
    if mode == "overhead":
        return mode, Vector((
            _uniform(rng, mode_config["x"], "camera.overhead.x"),
            _uniform(rng, mode_config["y"], "camera.overhead.y"),
            _uniform(rng, mode_config["z"], "camera.overhead.z"),
        )), None
    raise ValueError(f"unsupported camera mode in YAML: {mode}")

def _camera_accepts(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    *,
    min_base: int,
    min_total: int,
) -> bool:
    projected = [_project_point(scene, camera, Vector(point)) for point in KEYPOINT_WORLD]
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    inside = [depth > 0.0 and 0.0 <= x < width and 0.0 <= y < height for x, y, depth in projected]
    base_count = sum(inside[:10])
    total_count = sum(inside)
    return base_count >= min_base and total_count >= min_total


def _classify_camera_orientation(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
) -> dict[str, Any]:
    """Resolve the physical 8-way source and its 4-way Court36 training mode."""

    source_camera_position = classify_outside_camera_position(
        camera.location,
        court_width=COURT_WIDTH,
        court_length=COURT_LENGTH,
    )
    if source_camera_position is None:
        far_projection = _project_point(
            scene,
            camera,
            Vector((COURT_WIDTH * 0.5, 0.0, 0.04)),
        )
        near_projection = _project_point(
            scene,
            camera,
            Vector((COURT_WIDTH * 0.5, COURT_LENGTH, 0.04)),
        )
        if far_projection[2] <= 0.0 or near_projection[2] <= 0.0:
            raise RuntimeError(
                "cannot canonicalize an inside/overhead camera because the court long axis is not projectable"
            )
        source_camera_position = classify_projected_camera_position(
            far_projection[:2],
            near_projection[:2],
        )
    return {
        "source_camera_position": source_camera_position,
        "canonical_camera_mode": canonical_camera_mode(source_camera_position),
        "keypoint_permutation": list(
            keypoint_permutation(source_camera_position, KEYPOINT_WORLD)
        ),
    }

def _randomize_camera(state: dict[str, Any], rng: random.Random) -> str:
    scene = state["scene"]
    camera = state["camera"]
    config = state["config"]
    camera_config = config["camera"]
    requested_partial = rng.random() < float(camera_config["requested_partial_probability"])
    partial = requested_partial
    mode = "sideline"
    focus: Vector | None = None
    maximum_attempts = int(camera_config["maximum_sampling_attempts"])
    if maximum_attempts < 1:
        raise ValueError("camera.maximum_sampling_attempts must be positive")
    for _ in range(maximum_attempts):
        mode, location, focus = _sample_camera_location(
            rng,
            float(state["net_height"]),
            config,
            str(state.get("arena_profile", "championship")),
        )
        mode_config = camera_config["modes"][mode]
        camera.location = location
        camera.data.lens = _uniform(rng, mode_config["lens_mm"], f"camera.modes.{mode}.lens_mm")
        camera.data.sensor_width = _uniform(rng, camera_config["sensor_width_mm"], "camera.sensor_width_mm")
        shift_scale = float(mode_config["shift_scale"])
        camera.data.shift_x = rng.uniform(-shift_scale, shift_scale)
        shift_y_limit = shift_scale * float(camera_config["shift_y_ratio"])
        camera.data.shift_y = rng.uniform(-shift_y_limit, shift_y_limit)
        target_jitter = camera_config["target_jitter"]
        if focus is not None:
            target = focus
        elif mode == "championship_main":
            target = COURT_CENTER + Vector((
                _uniform(rng, target_jitter["championship_main_x"], "camera.target_jitter.championship_main_x"),
                _uniform(rng, target_jitter["championship_main_y"], "camera.target_jitter.championship_main_y"),
                _uniform(rng, target_jitter["championship_main_z"], "camera.target_jitter.championship_main_z"),
            ))
        elif mode == "broadcast_sideline":
            target = COURT_CENTER + Vector((
                _uniform(rng, target_jitter["broadcast_sideline_x"], "camera.target_jitter.broadcast_sideline_x"),
                _uniform(rng, target_jitter["broadcast_sideline_y"], "camera.target_jitter.broadcast_sideline_y"),
                _uniform(rng, target_jitter["broadcast_sideline_z"], "camera.target_jitter.broadcast_sideline_z"),
            ))
        else:
            target = COURT_CENTER + Vector((
                _uniform(rng, target_jitter["other_x"], "camera.target_jitter.other_x"),
                _uniform(rng, target_jitter["other_y"], "camera.target_jitter.other_y"),
                _uniform(rng, target_jitter["other_z"], "camera.target_jitter.other_z"),
            ))
        roll_degrees = _uniform(rng, mode_config["roll_degrees"], f"camera.modes.{mode}.roll_degrees")
        _look_at(camera, target, roll=math.radians(roll_degrees))
        bpy.context.view_layer.update()
        partial = (requested_partial and mode != "championship_main") or mode in {
            "server_wide",
            "server_closeup",
            "net_aligned_server_pan",
        }
        if partial and mode not in {"server_wide", "server_closeup", "net_aligned_server_pan"}:
            acceptance = camera_config["partial_acceptance"]
        else:
            acceptance = mode_config
        min_base = int(acceptance["min_base_keypoints"])
        min_total = int(acceptance["min_total_keypoints"])
        if _camera_accepts(scene, camera, min_base=min_base, min_total=min_total):
            break
    else:
        fallback = camera_config["fallback"]
        camera.location = Vector(fallback["location"])
        camera.data.lens = float(fallback["lens_mm"])
        camera.data.sensor_width = float(fallback["sensor_width_mm"])
        camera.data.shift_x = 0.0
        camera.data.shift_y = 0.0
        _look_at(camera, COURT_CENTER + Vector((0.0, 0.0, float(fallback["target_z"]))))
        mode = "fallback_sideline"
        focus = None
        partial = False
    bpy.context.view_layer.update()
    state.update(_classify_camera_orientation(scene, camera))
    state["server_focus"] = [float(focus.x), float(focus.y), 0.0] if focus is not None else None
    return f"{mode}{'_partial' if partial else ''}"

def _randomize_players(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    players: list[bpy.types.Object] = state["players"]
    camera: bpy.types.Object = state["camera"]
    config = state["config"]
    player_config = config["players"]
    server_focus = state.get("server_focus")
    if server_focus is not None:
        pattern = "server_with_clutter"
        pattern_config = player_config["patterns"][pattern]
    else:
        selectable_patterns = {
            name: entry
            for name, entry in player_config["patterns"].items()
            if not bool(entry.get("forced_by_server_camera", False))
        }
        pattern, pattern_config = _weighted_entry(rng, config, selectable_patterns, "players.patterns")
    count_low, count_high = _range_pair(pattern_config["count"], f"players.patterns.{pattern}.count")
    active_count = rng.randint(int(count_low), min(int(count_high), len(players)))
    forced_blockers = _randint(rng, player_config["forced_blocker_count"], "players.forced_blocker_count")
    sideline_cluster_side = -1 if rng.random() < float(player_config["sideline_cluster_left_probability"]) else 1
    sideline_config = player_config["sideline_cluster"]
    sideline_cluster_center = Vector((
        float(sideline_config["left_center_x"] if sideline_cluster_side < 0 else sideline_config["right_center_x"]),
        _uniform(rng, sideline_config["center_y"], "players.sideline_cluster.center_y"),
        0.0,
    ))
    for root in players:
        root.hide_render = True
        root.hide_viewport = True
        for child in root.children_recursive:
            child.hide_render = True
            child.hide_viewport = True
    for index, root in enumerate(players[:active_count]):
        root.hide_render = False
        root.hide_viewport = False
        for child in root.children_recursive:
            child.hide_render = False
            child.hide_viewport = False
        height_scale = _uniform(rng, player_config["height_scale"], "players.height_scale")
        root.scale = (height_scale, height_scale, height_scale)
        root.rotation_euler = (0.0, 0.0, math.radians(_uniform(rng, player_config["rotation_degrees"], "players.rotation_degrees")))
        if index == 0 and server_focus is not None:
            root.location = tuple(server_focus)
            root.rotation_euler = (0.0, 0.0, 0.0 if float(server_focus[1]) < 9.0 else math.pi)
        elif index < forced_blockers + (1 if server_focus is not None else 0):
            target = Vector(rng.choice(KEYPOINT_WORLD))
            desired_ray_height = _uniform(rng, player_config["blocker_ray_height"], "players.blocker_ray_height")
            alpha_low, alpha_high = _range_pair(player_config["blocker_alpha"], "players.blocker_alpha")
            alpha = min(alpha_high, max(alpha_low, desired_ray_height / max(1.0, camera.location.z)))
            location = target + (camera.location - target) * alpha
            root.location = (location.x, location.y, 0.0)
        elif pattern == "net_cluster" and rng.random() < float(player_config["net_cluster"]["probability"]):
            cluster = player_config["net_cluster"]
            clip_x = _range_pair(cluster["clip_x"], "players.net_cluster.clip_x")
            clip_y = _range_pair(cluster["clip_y"], "players.net_cluster.clip_y")
            root.location = (
                min(clip_x[1], max(clip_x[0], rng.gauss(float(cluster["center_x"]), float(cluster["sigma_x"])))),
                min(clip_y[1], max(clip_y[0], rng.gauss(float(cluster["center_y"]), float(cluster["sigma_y"])))),
                0.0,
            )
        elif pattern == "sideline_cluster" and rng.random() < float(sideline_config["probability"]):
            root.location = (
                rng.gauss(sideline_cluster_center.x, float(sideline_config["sigma_x"])),
                rng.gauss(sideline_cluster_center.y, float(sideline_config["sigma_y"])),
                0.0,
            )
        elif pattern == "two_team_huddles" and rng.random() < float(player_config["two_team_huddles"]["probability"]):
            huddles = player_config["two_team_huddles"]
            center_values = huddles["team_a_center"] if index % 2 == 0 else huddles["team_b_center"]
            center = Vector((float(center_values[0]), float(center_values[1]), 0.0))
            root.location = (
                rng.gauss(center.x, float(huddles["sigma_x"])),
                rng.gauss(center.y, float(huddles["sigma_y"])),
                0.0,
            )
        elif pattern == "server_with_clutter" and rng.random() < float(player_config["server_with_clutter"]["cluster_probability"]):
            server_clutter = player_config["server_with_clutter"]
            if rng.random() < float(server_clutter["net_cluster_probability"]):
                cluster_y = float(server_clutter["net_y"])
            else:
                baseline_y = server_clutter["baseline_y"]
                cluster_y = float(
                    baseline_y[0]
                    if rng.random() < float(server_clutter["far_baseline_probability"])
                    else baseline_y[1]
                )
            root.location = (
                _uniform(rng, server_clutter["x"], "players.server_with_clutter.x"),
                rng.gauss(cluster_y, float(server_clutter["sigma_y"])),
                0.0,
            )
        else:
            fallback = player_config["fallback"]
            if rng.random() < float(fallback["inside_probability"]):
                root.location = (
                    _uniform(rng, fallback["inside_x"], "players.fallback.inside_x"),
                    _uniform(rng, fallback["inside_y"], "players.fallback.inside_y"),
                    0.0,
                )
            else:
                side, _ = _weighted_entry(
                    rng,
                    config,
                    fallback["outside_sides"],
                    "players.fallback.outside_sides",
                )
                if side == "left":
                    root.location = (_uniform(rng, fallback["left_x"], "players.fallback.left_x"), _uniform(rng, fallback["side_y"], "players.fallback.side_y"), 0.0)
                elif side == "right":
                    root.location = (_uniform(rng, fallback["right_x"], "players.fallback.right_x"), _uniform(rng, fallback["side_y"], "players.fallback.side_y"), 0.0)
                elif side == "far":
                    root.location = (_uniform(rng, fallback["far_x"], "players.fallback.far_x"), _uniform(rng, fallback["far_y"], "players.fallback.far_y"), 0.0)
                else:
                    root.location = (_uniform(rng, fallback["near_x"], "players.fallback.near_x"), _uniform(rng, fallback["near_y"], "players.fallback.near_y"), 0.0)
    return {"player_count": active_count, "player_pattern": pattern}

def _randomize_props(state: dict[str, Any], rng: random.Random) -> int:
    props: list[bpy.types.Object] = state["props"]
    config = state["config"]
    prop_config = config["distractions"]
    active_low, active_high = _range_pair(prop_config["active_count"], "distractions.active_count")
    active_count = rng.randint(int(active_low), min(int(active_high), len(props)))
    for index, prop in enumerate(props):
        active = index < active_count
        prop.hide_render = not active
        prop.hide_viewport = not active
        if not active:
            continue
        if rng.random() < float(prop_config["inside_probability"]):
            side = "inside"
        else:
            side, _ = _weighted_entry(
                rng,
                config,
                prop_config["outside_sides"],
                "distractions.outside_sides",
            )
        if side == "inside":
            inside = prop_config["inside"]
            prop.location = (
                _uniform(rng, inside["x"], "distractions.inside.x"),
                _uniform(rng, inside["y"], "distractions.inside.y"),
                _uniform(rng, inside["z"], "distractions.inside.z"),
            )
            prop.rotation_euler = (0.0, 0.0, math.radians(_uniform(rng, inside["rotation_degrees"], "distractions.inside.rotation_degrees")))
            prop.dimensions = (
                _uniform(rng, inside["size_x"], "distractions.inside.size_x"),
                _uniform(rng, inside["size_y"], "distractions.inside.size_y"),
                _uniform(rng, inside["size_z"], "distractions.inside.size_z"),
            )
            continue
        outside = prop_config["outside"]
        if side == "left":
            prop.location = (_uniform(rng, outside["left_x"], "distractions.outside.left_x"), _uniform(rng, outside["side_y"], "distractions.outside.side_y"), _uniform(rng, outside["z"], "distractions.outside.z"))
            prop.rotation_euler = (0.0, 0.0, math.radians(90.0 + _uniform(rng, outside["rotation_degrees"], "distractions.outside.rotation_degrees")))
        elif side == "right":
            prop.location = (_uniform(rng, outside["right_x"], "distractions.outside.right_x"), _uniform(rng, outside["side_y"], "distractions.outside.side_y"), _uniform(rng, outside["z"], "distractions.outside.z"))
            prop.rotation_euler = (0.0, 0.0, math.radians(90.0 + _uniform(rng, outside["rotation_degrees"], "distractions.outside.rotation_degrees")))
        elif side == "far":
            prop.location = (_uniform(rng, outside["far_x"], "distractions.outside.far_x"), _uniform(rng, outside["far_y"], "distractions.outside.far_y"), _uniform(rng, outside["z"], "distractions.outside.z"))
            prop.rotation_euler = (0.0, 0.0, math.radians(_uniform(rng, outside["rotation_degrees"], "distractions.outside.rotation_degrees")))
        else:
            prop.location = (_uniform(rng, outside["near_x"], "distractions.outside.near_x"), _uniform(rng, outside["near_y"], "distractions.outside.near_y"), _uniform(rng, outside["z"], "distractions.outside.z"))
            prop.rotation_euler = (0.0, 0.0, math.radians(_uniform(rng, outside["rotation_degrees"], "distractions.outside.rotation_degrees")))
        prop.dimensions = (
            _uniform(rng, outside["size_x"], "distractions.outside.size_x"),
            _uniform(rng, outside["size_y"], "distractions.outside.size_y"),
            _uniform(rng, outside["size_z"], "distractions.outside.size_z"),
        )
    return active_count

def _set_hierarchy_visibility(root: bpy.types.Object, visible: bool) -> None:
    root.hide_render = not visible
    root.hide_viewport = not visible
    for child in root.children_recursive:
        child.hide_render = not visible
        child.hide_viewport = not visible

def _randomize_arena_background(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    config = state["config"]
    arena_config = config["arena"]
    forced_profile = str(arena_config.get("forced_profile", "")).strip()
    profile_sequence = state.get("arena_profile_sequence")
    if profile_sequence is not None:
        if not isinstance(profile_sequence, (list, tuple)) or not profile_sequence:
            raise ValueError("arena_profile_sequence must contain at least one arena profile")
        sample_index = int(state.get("sample_index", 0))
        profile = str(profile_sequence[sample_index % len(profile_sequence)])
        if profile not in arena_config["profiles"]:
            raise ValueError(f"scheduled arena profile is not defined in arena.profiles: {profile}")
    elif forced_profile:
        if forced_profile not in arena_config["profiles"]:
            raise ValueError(f"arena.forced_profile is not defined in arena.profiles: {forced_profile}")
        profile = forced_profile
    else:
        profile, _ = _weighted_entry(rng, config, arena_config["profiles"], "arena.profiles")
    sky_sphere = randomize_sky_sphere(state, rng, profile)
    if profile == "small_gym":
        active_side = (
            0
            if rng.random() < float(arena_config["small_gym_bilateral_probability"])
            else (-1 if rng.random() < float(arena_config["left_side_probability"]) else 1)
        )
    elif profile in {"none", "sky_sphere"}:
        active_side = 0
    else:
        active_side = -1 if rng.random() < float(arena_config["left_side_probability"]) else 1
    background_count = 0
    for obj in state["arena_backgrounds"]:
        object_profile = str(obj.get("arena_profile", ""))
        object_side = int(obj.get("arena_side", 0))
        visible = object_profile == profile
        if visible and profile == "small_gym" and active_side != 0 and object_side not in (0, active_side):
            visible = False
        if visible and obj.get("arena_led") and rng.random() < float(arena_config["led_dropout_probability"]):
            visible = False
        if visible and obj.get("arena_led") and obj.data is not None and obj.data.materials:
            atlas_materials = state["materials"].get("led_atlas", ())
            if atlas_materials:
                _, variant = _weighted_entry(
                    rng,
                    config,
                    arena_config["advertising_variants"],
                    "arena.advertising_variants",
                )
                obj.data.materials[0] = atlas_materials[int(variant["index"]) % len(atlas_materials)]
        obj.hide_render = not visible
        obj.hide_viewport = not visible
        background_count += int(visible)

    # A panorama already owns the entire arena shell. Keep every generated LED
    # board hidden so it cannot float in front of the sky-sphere audience.
    if profile == "sky_sphere":
        for led_board in state["led_boards"]:
            led_board.hide_render = True
            led_board.hide_viewport = True
    advertising_led_count = sum(int(not obj.hide_render) for obj in state["led_boards"])

    audience_config = arena_config["audience"]
    if profile == "championship":
        audience_density = _uniform(rng, audience_config["championship_density"], "arena.audience.championship_density")
    elif profile == "yellow_grandstand":
        audience_density = _uniform(rng, audience_config["yellow_grandstand_density"], "arena.audience.yellow_grandstand_density")
    elif profile == "small_gym":
        audience_density = _uniform(rng, audience_config["small_gym_density"], "arena.audience.small_gym_density")
    else:
        audience_density = 0.0
    audience_count = 0
    for root in state["audience"]:
        root_profile = str(root.get("arena_profile", ""))
        root_side = int(root.get("arena_side", 0))
        visible = root_profile == profile and rng.random() < audience_density
        if visible and profile == "small_gym" and active_side != 0 and root_side != active_side:
            visible = False
        _set_hierarchy_visibility(root, visible)
        if not visible:
            continue
        scale = _uniform(rng, audience_config["scale"], "arena.audience.scale")
        root.scale = (scale, scale, scale)
        base_rotation = 90.0 if root_side < 0 else -90.0
        rotation_offset = _uniform(rng, audience_config["rotation_degrees"], "arena.audience.rotation_degrees")
        root.rotation_euler = (0.0, 0.0, math.radians(base_rotation + rotation_offset))
        audience_count += 1

    state["arena_profile"] = profile
    state["arena_side"] = active_side
    return {
        "arena_profile": profile,
        "arena_side": active_side,
        "audience_count": audience_count,
        "arena_background_count": background_count,
        "advertising_led_count": advertising_led_count,
        **sky_sphere,
    }

def _randomize_lighting(state: dict[str, Any], rng: random.Random) -> dict[str, float]:
    scene = state["scene"]
    lighting_config = state["config"]["lighting"]
    sun = bpy.data.objects.get("LIGHT_sun")
    sun.data.energy = _uniform(rng, lighting_config["sun_energy"], "lighting.sun_energy")
    # Draw each axis independently while keeping one shared configurable range.
    sun.rotation_euler = tuple(
        math.radians(_uniform(rng, lighting_config["sun_rotation_degrees"], "lighting.sun_rotation_degrees"))
        for _ in range(3)
    )
    for index in range(4):
        light = bpy.data.objects.get(f"LIGHT_area_{index}")
        light.data.energy = _uniform(rng, lighting_config["area_light_energy"], "lighting.area_light_energy")
        light.data.color = (
            _uniform(rng, lighting_config["area_light_red"], "lighting.area_light_red"),
            _uniform(rng, lighting_config["area_light_green"], "lighting.area_light_green"),
            _uniform(rng, lighting_config["area_light_blue"], "lighting.area_light_blue"),
        )
    background = scene.world.node_tree.nodes.get("Background")
    world_strength = _uniform(rng, lighting_config["world_strength"], "lighting.world_strength")
    world_strength *= float(state.get("sky_sphere_strength_multiplier", 1.0))
    background.inputs["Strength"].default_value = world_strength
    scene.view_settings.exposure = _uniform(rng, lighting_config["exposure"], "lighting.exposure")
    return {"sun_energy": float(sun.data.energy), "world_strength": float(background.inputs["Strength"].default_value)}

def randomize_scene(state: dict[str, Any], sample_seed: int) -> dict[str, Any]:
    rng = random.Random(sample_seed)
    palette = _randomize_materials(state, rng)
    arena = _randomize_arena_background(state, rng)
    camera_mode = _randomize_camera(state, rng)
    player_info = _randomize_players(state, rng)
    distraction_count = _randomize_props(state, rng)
    lighting = _randomize_lighting(state, rng)
    bpy.context.view_layer.update()
    return {
        "palette": palette,
        "camera_mode": camera_mode,
        "source_camera_position": state["source_camera_position"],
        "canonical_camera_mode": state["canonical_camera_mode"],
        "keypoint_permutation": list(state["keypoint_permutation"]),
        **player_info,
        "distraction_count": distraction_count,
        **arena,
        **lighting,
    }
