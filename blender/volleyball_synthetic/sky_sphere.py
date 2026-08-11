"""Blender World sky-sphere support for synthetic volleyball arenas.

This module intentionally owns panorama loading, World shader nodes, profile
activation, rotation, and environment-strength metadata. The dataset generator
only decides when the ``sky_sphere`` arena profile is active.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import bpy


def _sample_range(rng: random.Random, value: Any, name: str) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed maximum")
    return rng.uniform(low, high)


def _entry_weight(config: dict[str, Any], entry: dict[str, Any], name: str) -> float:
    if str(config.get("selection_weight_mode", "exact")) == "stars":
        stars = str(entry.get("stars", ""))
        star_weights = config.get("star_weights", {})
        if stars not in star_weights:
            raise ValueError(f"unknown stars value for {name}: {stars!r}")
        return float(star_weights[stars])
    return float(entry.get("weight", 0.0))


def _choose_asset(
    rng: random.Random,
    config: dict[str, Any],
    entries: dict[str, Any],
) -> str:
    if not isinstance(entries, dict) or not entries:
        raise ValueError("arena.sky_sphere.assets must contain at least one panorama")
    names = list(entries)
    weights = [
        _entry_weight(config, entries[name], f"arena.sky_sphere.assets.{name}")
        for name in names
    ]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("arena.sky_sphere.assets weights must have at least one positive value")
    return rng.choices(names, weights=weights, k=1)[0]


def build_sky_sphere_world(scene: bpy.types.Scene, config: dict[str, Any]) -> dict[str, Any]:
    """Prepare reusable equirectangular World nodes and load configured panoramas."""

    sky_config = config["arena"].get("sky_sphere", {})
    asset_entries = sky_config.get("assets", {})
    if not isinstance(asset_entries, dict) or not asset_entries:
        raise ValueError("arena.sky_sphere.assets must contain at least one panorama")

    config_directory = Path(config["_config_path"]).resolve().parent
    images: dict[str, bpy.types.Image] = {}
    image_paths: dict[str, str] = {}
    for asset_name, entry in asset_entries.items():
        raw_path = Path(str(entry.get("path", "")))
        asset_path = raw_path if raw_path.is_absolute() else config_directory / raw_path
        asset_path = asset_path.resolve()
        if not asset_path.exists():
            raise FileNotFoundError(f"sky sphere panorama does not exist: {asset_path}")
        image = bpy.data.images.load(str(asset_path), check_existing=True)
        image.colorspace_settings.name = "sRGB"
        images[str(asset_name)] = image
        image_paths[str(asset_name)] = str(asset_path)

    scene.world.use_nodes = True
    node_tree = scene.world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    background = nodes.get("Background")
    output = nodes.get("World Output")
    if background is None:
        background = nodes.new("ShaderNodeBackground")
        background.name = "Background"
    if output is None:
        output = nodes.new("ShaderNodeOutputWorld")
        output.name = "World Output"
    if not output.inputs["Surface"].is_linked:
        links.new(background.outputs["Background"], output.inputs["Surface"])

    texture_coordinate = nodes.new("ShaderNodeTexCoord")
    texture_coordinate.name = "SKYSPHERE_coordinates"
    mapping = nodes.new("ShaderNodeMapping")
    mapping.name = "SKYSPHERE_rotation"
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.name = "SKYSPHERE_environment"
    environment.projection = "EQUIRECTANGULAR"
    environment.interpolation = "Linear"
    # World shaders need a ray direction. Object-style Generated coordinates
    # collapse to an unusable constant here and sample one dark panorama pixel.
    links.new(texture_coordinate.outputs["Normal"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])

    return {
        "background": background,
        "environment": environment,
        "mapping": mapping,
        "images": images,
        "image_paths": image_paths,
        "default_color": tuple(background.inputs["Color"].default_value),
    }


def randomize_sky_sphere(
    state: dict[str, Any],
    rng: random.Random,
    profile: str,
) -> dict[str, Any]:
    """Enable one configured panorama only for the sky-sphere arena profile."""

    sky_state = state["sky_sphere"]
    background = sky_state["background"]
    node_tree = state["scene"].world.node_tree
    for link in list(background.inputs["Color"].links):
        node_tree.links.remove(link)
    background.inputs["Color"].default_value = sky_state["default_color"]
    state["sky_sphere_strength_multiplier"] = 1.0
    state["sky_sphere_active"] = False

    if profile != "sky_sphere":
        return {
            "sky_sphere_asset": "none",
            "sky_sphere_rotation_degrees": 0.0,
            "sky_sphere_strength_multiplier": 1.0,
        }

    config = state["config"]
    sky_config = config["arena"]["sky_sphere"]
    sample_index = int(state.get("sample_index", 0))
    asset_by_sample_index = state.get("sky_sphere_asset_by_sample_index")
    asset_sequence = state.get("sky_sphere_asset_sequence")
    if asset_by_sample_index is not None:
        if not isinstance(asset_by_sample_index, (list, tuple)):
            raise ValueError("sky_sphere_asset_by_sample_index must be a list or tuple")
        if sample_index < 0 or sample_index >= len(asset_by_sample_index):
            raise ValueError(
                f"sky sphere sample index {sample_index} is outside the scheduled asset list"
            )
        asset_name = str(asset_by_sample_index[sample_index] or "").strip()
        if not asset_name:
            raise ValueError(f"sky sphere sample index {sample_index} has no scheduled asset")
        if asset_name not in sky_config["assets"]:
            raise ValueError(f"scheduled sky sphere asset is not configured: {asset_name}")
    elif asset_sequence is not None:
        if not isinstance(asset_sequence, (list, tuple)) or not asset_sequence:
            raise ValueError("sky_sphere_asset_sequence must contain at least one asset name")
        asset_name = str(asset_sequence[sample_index % len(asset_sequence)])
        if asset_name not in sky_config["assets"]:
            raise ValueError(f"scheduled sky sphere asset is not configured: {asset_name}")
    else:
        asset_name = _choose_asset(rng, config, sky_config["assets"])
    asset_config = sky_config["assets"][asset_name]
    rotation_degrees = _sample_range(
        rng,
        asset_config.get("rotation_degrees", sky_config["rotation_degrees"]),
        "arena.sky_sphere.rotation_degrees",
    )
    yaw_offset_degrees = float(asset_config.get("yaw_offset_degrees", 0.0))
    orientation = asset_config.get(
        "orientation_degrees",
        sky_config.get("orientation_degrees", [180.0, 0.0, 0.0]),
    )
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 3:
        raise ValueError("arena.sky_sphere.orientation_degrees must contain three values")
    mapping_scale = asset_config.get("mapping_scale", [1.0, 1.0, 1.0])
    mapping_location = asset_config.get("mapping_location", [0.0, 0.0, 0.0])
    if not isinstance(mapping_scale, (list, tuple)) or len(mapping_scale) != 3:
        raise ValueError(f"{asset_name}.mapping_scale must contain three values")
    if not isinstance(mapping_location, (list, tuple)) or len(mapping_location) != 3:
        raise ValueError(f"{asset_name}.mapping_location must contain three values")
    strength_multiplier = _sample_range(
        rng,
        asset_config.get("strength_multiplier", sky_config["strength_multiplier"]),
        "arena.sky_sphere.strength_multiplier",
    )
    sky_state["environment"].image = sky_state["images"][asset_name]
    sky_state["mapping"].inputs["Location"].default_value = tuple(
        float(value) for value in mapping_location
    )
    sky_state["mapping"].inputs["Rotation"].default_value = (
        math.radians(float(orientation[0])),
        math.radians(float(orientation[1])),
        math.radians(float(orientation[2]) + yaw_offset_degrees + rotation_degrees),
    )
    sky_state["mapping"].inputs["Scale"].default_value = tuple(
        float(value) for value in mapping_scale
    )
    node_tree.links.new(sky_state["environment"].outputs["Color"], background.inputs["Color"])
    state["sky_sphere_strength_multiplier"] = strength_multiplier
    state["sky_sphere_active"] = True
    return {
        "sky_sphere_asset": asset_name,
        "sky_sphere_path": sky_state["image_paths"][asset_name],
        "sky_sphere_rotation_degrees": float(rotation_degrees),
        "sky_sphere_yaw_offset_degrees": yaw_offset_degrees,
        "sky_sphere_mapping_scale": [float(value) for value in mapping_scale],
        "sky_sphere_mapping_location": [float(value) for value in mapping_location],
        "sky_sphere_source_floor_boundary_uv": float(
            asset_config.get("source_floor_boundary_uv", 0.5)
        ),
        "sky_sphere_arena_type": str(asset_config.get("arena_type", "unknown")),
        "sky_sphere_strength_multiplier": float(strength_multiplier),
    }
