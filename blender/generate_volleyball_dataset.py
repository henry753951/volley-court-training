"""Procedural Blender generator for the canonical 36-point volleyball court.

This file is intentionally self-contained so it can be executed by Blender's
bundled Python, either from the command line or through the official Blender
Lab MCP add-on.

Coordinate convention (metres):

* X: court width, left (0) to right (9)
* Y: court length, far baseline (0) to near baseline (18)
* Z: height above the playing surface

YOLO visibility follows the Ultralytics pose convention: 0 is outside or
unknown, 1 is located but occluded, and 2 is visible.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector


COURT_WIDTH = 9.0
COURT_LENGTH = 18.0
COURT_CENTER = Vector((COURT_WIDTH / 2.0, COURT_LENGTH / 2.0, 0.0))
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "randomization-config.yaml"

KEYPOINT_NAMES = (
    "court_left_near_corner",
    "court_left_attack_near",
    "court_left_net_line",
    "court_left_attack_far",
    "court_left_far_corner",
    "court_right_far_corner",
    "court_right_attack_far",
    "court_right_net_line",
    "court_right_attack_near",
    "court_right_near_corner",
    "court_left_near_sideline_third_1",
    "court_left_near_sideline_third_2",
    "court_left_near_attack_zone_third_1",
    "court_left_near_attack_zone_third_2",
    "court_left_far_attack_zone_third_1",
    "court_left_far_attack_zone_third_2",
    "court_left_far_sideline_third_1",
    "court_left_far_sideline_third_2",
    "court_far_baseline_third_1",
    "court_far_baseline_third_2",
    "court_right_far_sideline_third_1",
    "court_right_far_sideline_third_2",
    "court_right_far_attack_zone_third_1",
    "court_right_far_attack_zone_third_2",
    "court_right_near_attack_zone_third_1",
    "court_right_near_attack_zone_third_2",
    "court_right_near_sideline_third_1",
    "court_right_near_sideline_third_2",
    "court_near_baseline_third_1",
    "court_near_baseline_third_2",
    "court_near_attack_line_third_1",
    "court_near_attack_line_third_2",
    "court_center_line_third_1",
    "court_center_line_third_2",
    "court_far_attack_line_third_1",
    "court_far_attack_line_third_2",
)

BASE_POINTS = (
    (0.0, 18.0, 0.04),
    (0.0, 12.0, 0.04),
    (0.0, 9.0, 0.04),
    (0.0, 6.0, 0.04),
    (0.0, 0.0, 0.04),
    (9.0, 0.0, 0.04),
    (9.0, 6.0, 0.04),
    (9.0, 9.0, 0.04),
    (9.0, 12.0, 0.04),
    (9.0, 18.0, 0.04),
)

BASE_SEGMENTS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
    (1, 8),
    (2, 7),
    (3, 6),
)


def _lerp_point(start: tuple[float, float, float], end: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(float(a + (b - a) * t) for a, b in zip(start, end))


KEYPOINT_WORLD = BASE_POINTS + tuple(
    point
    for start_index, end_index in BASE_SEGMENTS
    for point in (
        _lerp_point(BASE_POINTS[start_index], BASE_POINTS[end_index], 1.0 / 3.0),
        _lerp_point(BASE_POINTS[start_index], BASE_POINTS[end_index], 2.0 / 3.0),
    )
)

if len(KEYPOINT_NAMES) != 36 or len(KEYPOINT_WORLD) != 36:
    raise RuntimeError("the synthetic court must preserve exactly 36 keypoints")


def _strip_yaml_comment(raw_line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(raw_line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None:
            return raw_line[:index]
    return raw_line


def _parse_yaml_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith(("[", "'", '"')):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"unsupported YAML scalar: {value}") from exc
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_yaml_key(raw_key: str) -> str:
    key = raw_key.strip()
    if key.startswith(("'", '"')):
        try:
            parsed = ast.literal_eval(key)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"unsupported quoted YAML key: {key}") from exc
        if not isinstance(parsed, str):
            raise ValueError(f"YAML mapping key must be a string: {key}")
        return parsed
    return key


def _load_yaml_without_dependency(path: Path) -> dict[str, Any]:
    """Parse the mapping-only YAML subset used by randomization-config.yaml."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "\t" in raw_line:
            raise ValueError(f"tabs are not supported in {path}:{line_number}")
        without_comment = _strip_yaml_comment(raw_line).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        content = without_comment.strip()
        if ":" not in content:
            raise ValueError(f"expected key: value in {path}:{line_number}: {content}")
        raw_key, raw_value = content.split(":", 1)
        key_value = _parse_yaml_key(raw_key)
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"mapping key must be a non-empty string in {path}:{line_number}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if key_value in parent:
            raise ValueError(f"duplicate key {key_value!r} in {path}:{line_number}")
        if raw_value.strip():
            parent[key_value] = _parse_yaml_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key_value] = child
            stack.append((indent, child))
    return root


def load_randomization_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"randomization config does not exist: {config_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        config = _load_yaml_without_dependency(config_path)
        parser_name = "builtin-yaml-subset"
    else:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"randomization config root must be a mapping: {config_path}")
        config = loaded
        parser_name = "PyYAML"
    required_sections = {
        "generation",
        "splits",
        "scene",
        "materials",
        "arena",
        "camera",
        "players",
        "distractions",
        "lighting",
        "capture",
    }
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"randomization config is missing sections: {', '.join(missing)}")
    mode = str(config.get("selection_weight_mode", "exact"))
    if mode not in {"exact", "stars"}:
        raise ValueError("selection_weight_mode must be 'exact' or 'stars'")

    def validate_probabilities(node: Any, prefix: str = "") -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            path_name = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_probability"):
                probability = float(value)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{path_name} must be between 0 and 1")
            validate_probabilities(value, path_name)

    validate_probabilities(config)
    config["_config_path"] = str(config_path)
    config["_yaml_parser"] = parser_name
    return config


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed maximum")
    return low, high


def _uniform(rng: random.Random, value: Any, name: str) -> float:
    low, high = _range_pair(value, name)
    return rng.uniform(low, high)


def _randint(rng: random.Random, value: Any, name: str) -> int:
    low, high = _range_pair(value, name)
    if not low.is_integer() or not high.is_integer():
        raise ValueError(f"{name} must contain integer bounds")
    return rng.randint(int(low), int(high))


def _entry_weight(config: dict[str, Any], entry: dict[str, Any], name: str) -> float:
    mode = str(config.get("selection_weight_mode", "exact"))
    if mode == "stars":
        stars = str(entry.get("stars", ""))
        star_weights = config.get("star_weights", {})
        if stars not in star_weights:
            raise ValueError(f"unknown stars value for {name}: {stars!r}")
        return float(star_weights[stars])
    return float(entry.get("weight", 0.0))


def _weighted_entry(
    rng: random.Random,
    config: dict[str, Any],
    entries: dict[str, Any],
    name: str,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{name} must contain at least one entry")
    names = list(entries)
    rows = [entries[item_name] for item_name in names]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{name} entries must be mappings")
    weights = [_entry_weight(config, row, f"{name}.{item_name}") for item_name, row in zip(names, rows)]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"{name} weights must be non-negative with at least one positive value")
    selected_name = rng.choices(names, weights=weights, k=1)[0]
    return selected_name, entries[selected_name]


def _collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in tuple(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _set_principled_input(material: bpy.types.Material, name: str, value: Any) -> None:
    node = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if node is not None and name in node.inputs:
        node.inputs[name].default_value = value


def _noisy_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    noise_scale: float = 7.0,
    noise_strength: float = 0.16,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = "NoiseCoordinates"
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "SurfaceNoise"
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.65
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "NoiseRamp"
    low = tuple(max(0.0, channel * (1.0 - noise_strength)) for channel in color[:3]) + (1.0,)
    high = tuple(min(1.0, channel * (1.0 + noise_strength)) for channel in color[:3]) + (1.0,)
    ramp.color_ramp.elements[0].color = low
    ramp.color_ramp.elements[1].color = high
    links.new(texcoord.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.10
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = min(0.45, roughness + 0.08)
    material.diffuse_color = color
    return material


def _simple_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.38,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    _set_principled_input(material, "Base Color", color)
    _set_principled_input(material, "Roughness", roughness)
    _set_principled_input(material, "Metallic", metallic)
    return material


def _emissive_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    strength: float = 2.0,
) -> bpy.types.Material:
    material = _simple_material(name, color, roughness=0.28)
    _set_principled_input(material, "Emission Color", color)
    _set_principled_input(material, "Emission Strength", strength)
    return material


def _advertising_atlas_materials() -> tuple[bpy.types.Material, ...]:
    """Create UV-shifted LED materials from the packed fictional sponsor atlas."""

    atlas_path = Path(__file__).resolve().parent / "assets" / "fictional-sponsor-led-atlas-v1.png"
    if not atlas_path.exists():
        return tuple(
            _emissive_material(
                f"MAT_LED_Fallback_{index}",
                ((0.04, 0.15, 0.95, 1.0), (0.95, 0.96, 1.0, 1.0), (0.95, 0.04, 0.06, 1.0))[index % 3],
                strength=2.2,
            )
            for index in range(6)
        )

    image = bpy.data.images.load(str(atlas_path), check_existing=True)
    image.name = "TEX_FictionalSponsorLEDAtlas"
    image.pack()
    materials: list[bpy.types.Material] = []
    for index in range(6):
        material = bpy.data.materials.new(f"MAT_LED_Atlas_{index}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        texcoord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.vector_type = "POINT"
        mapping.inputs["Location"].default_value = (index / 6.0, 0.0, 0.0)
        mapping.inputs["Scale"].default_value = (1.0 / 6.0, 1.0, 1.0)
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "REPEAT"
        links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        if "Emission Color" in bsdf.inputs:
            links.new(texture.outputs["Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.8
        bsdf.inputs["Roughness"].default_value = 0.28
        materials.append(material)
    return tuple(materials)


def _mask_image(name: str, width: int, height: int, mask: np.ndarray) -> bpy.types.Image:
    image = bpy.data.images.get(name)
    # Packed generated images retain their previous packed payload when reused.
    # Recreate them so a rebuilt .blend always receives the current mask pixels.
    if image is not None:
        bpy.data.images.remove(image)
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=False)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = mask[:, :, None]
    rgba[:, :, 3] = mask
    # In Blender 5.2 changing colorspace after filling a generated image resets
    # its pixels to the default black/opaque buffer. Configure first, then fill.
    image.alpha_mode = "STRAIGHT"
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(rgba.reshape(-1))
    image.update()
    image.pack()
    return image


def _court_line_mask(width: int = 1024, height: int = 2048) -> bpy.types.Image:
    mask = np.zeros((height, width), dtype=np.float32)
    line_px_x = max(3, round(width * 0.055 / COURT_WIDTH))
    line_px_y = max(3, round(height * 0.055 / COURT_LENGTH))

    def vertical(x: int) -> None:
        mask[:, max(0, x - line_px_x // 2) : min(width, x + line_px_x // 2 + 1)] = 1.0

    def horizontal(y: int) -> None:
        mask[max(0, y - line_px_y // 2) : min(height, y + line_px_y // 2 + 1), :] = 1.0

    vertical(0)
    vertical(width - 1)
    for fraction in (0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0):
        horizontal(round((height - 1) * fraction))
    return _mask_image("TEX_CourtLineMask", width, height, mask)


def _surround_decal_mask(width: int = 1536, height: int = 2048) -> bpy.types.Image:
    """White regulation marks and generic event lettering for the free zone."""

    mask = np.zeros((height, width), dtype=np.float32)
    x_min, x_max = -9.0, 18.0
    y_min, y_max = -9.0, 27.0

    def px_x(world_x: float) -> int:
        return round((world_x - x_min) / (x_max - x_min) * (width - 1))

    def px_y(world_y: float) -> int:
        return round((world_y - y_min) / (y_max - y_min) * (height - 1))

    def rectangle(x0: float, x1: float, y0: float, y1: float) -> None:
        left, right = sorted((px_x(x0), px_x(x1)))
        bottom, top = sorted((px_y(y0), px_y(y1)))
        mask[max(0, bottom) : min(height, top + 1), max(0, left) : min(width, right + 1)] = 1.0

    # Five dashed attack-line extensions on both sidelines, plus the short
    # service-zone marks behind both baselines.
    for attack_y in (6.0, 12.0):
        for index in range(5):
            start = 0.20 + index * 0.35
            rectangle(-start - 0.15, -start, attack_y - 0.0275, attack_y + 0.0275)
            rectangle(COURT_WIDTH + start, COURT_WIDTH + start + 0.15, attack_y - 0.0275, attack_y + 0.0275)
    for service_x in (0.0, COURT_WIDTH):
        rectangle(service_x - 0.0275, service_x + 0.0275, -0.35, -0.20)
        rectangle(service_x - 0.0275, service_x + 0.0275, COURT_LENGTH + 0.20, COURT_LENGTH + 0.35)

    font = {
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
        "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
        "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
        "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
        "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    }

    def draw_text(text: str, world_x: float, world_y: float, cell: int = 8) -> None:
        cursor_x = px_x(world_x)
        origin_y = px_y(world_y)
        for character in text:
            glyph = font.get(character)
            if glyph is not None:
                for row, bits in enumerate(glyph):
                    for column, bit in enumerate(bits):
                        if bit == "1":
                            x0 = cursor_x + column * cell
                            y0 = origin_y + (6 - row) * cell
                            mask[y0 : min(height, y0 + cell), x0 : min(width, x0 + cell)] = 1.0
            cursor_x += 6 * cell

    # Generic text avoids copying a real event logo while reproducing the
    # strong white floor typography seen in official arenas.
    draw_text("VOLLEYBALL", 0.70, -4.6, cell=8)
    draw_text("CHAMPIONSHIP", -0.55, -6.0, cell=7)
    draw_text("VOLLEYBALL", 0.70, 21.1, cell=8)
    draw_text("CHAMPIONSHIP", -0.55, 22.5, cell=7)
    return _mask_image("TEX_SurroundDecals", width, height, mask)


def _net_alpha_mask(width: int = 1024, height: int = 256) -> bpy.types.Image:
    mask = np.zeros((height, width), dtype=np.float32)
    # A broadcast camera and video compression make regulation-thin threads
    # disappear very easily.  Keep the physical occlusion test independent,
    # but render a slightly thicker photographic alpha mask so the grid stays
    # readable without turning the net into a solid plane.
    wire_x = 1
    wire_z = 1
    for column in range(91):
        x = round((width - 1) * column / 90.0)
        mask[:, max(0, x - wire_x) : min(width, x + wire_x + 1)] = 1.0
    for row in range(11):
        y = round((height - 1) * row / 10.0)
        mask[max(0, y - wire_z) : min(height, y + wire_z + 1), :] = 1.0
    return _mask_image("TEX_NetAlphaGrid", width, height, mask)


def _court_surface_material(
    color: tuple[float, float, float, float],
    line_color: tuple[float, float, float, float],
) -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Court")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    texcoord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "SurfaceNoise"
    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.62
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "NoiseRamp"
    ramp.color_ramp.elements[0].color = tuple(channel * 0.87 for channel in color[:3]) + (1.0,)
    ramp.color_ramp.elements[1].color = tuple(min(1.0, channel * 1.13) for channel in color[:3]) + (1.0,)
    line_texture = nodes.new("ShaderNodeTexImage")
    line_texture.name = "CourtLineMask"
    line_texture.image = _court_line_mask()
    line_texture.interpolation = "Linear"
    line_texture.extension = "CLIP"
    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = "CourtLineMix"
    mix.blend_type = "MIX"
    mix.inputs[2].default_value = line_color
    links.new(texcoord.outputs["Generated"], noise.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], line_texture.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(line_texture.outputs["Color"], mix.inputs[0])
    links.new(ramp.outputs["Color"], mix.inputs[1])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.18
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.24
    material.diffuse_color = color
    return material


def _surround_surface_material(
    color: tuple[float, float, float, float],
    decal_color: tuple[float, float, float, float],
) -> bpy.types.Material:
    material = _noisy_material(
        "MAT_Surround",
        color,
        roughness=0.22,
        noise_scale=6.0,
        noise_strength=0.18,
    )
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    texcoord = nodes.get("NoiseCoordinates")
    ramp = nodes.get("NoiseRamp")
    decal = nodes.new("ShaderNodeTexImage")
    decal.name = "SurroundDecalMask"
    decal.image = _surround_decal_mask()
    decal.interpolation = "Linear"
    decal.extension = "CLIP"
    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = "SurroundDecalMix"
    mix.blend_type = "MIX"
    mix.inputs[2].default_value = decal_color
    for link in tuple(links):
        if link.to_node == bsdf and link.to_socket.name == "Base Color":
            links.remove(link)
    links.new(texcoord.outputs["Generated"], decal.inputs["Vector"])
    links.new(decal.outputs["Color"], mix.inputs[0])
    links.new(ramp.outputs["Color"], mix.inputs[1])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def _net_texture_material() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_NetTexture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    texcoord = nodes.new("ShaderNodeTexCoord")
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "NetAlphaMask"
    texture.image = _net_alpha_mask()
    texture.interpolation = "Linear"
    texture.extension = "CLIP"
    bsdf.inputs["Base Color"].default_value = (0.035, 0.04, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    output = nodes.get("Material Output")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    for link in tuple(links):
        if link.to_node == output and link.to_socket.name == "Surface":
            links.remove(link)
    links.new(texcoord.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    material.diffuse_color = (0.035, 0.04, 0.05, 1.0)
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
    elif hasattr(material, "blend_method"):
        material.blend_method = "HASHED"
    return material


def _add_box(
    name: str,
    location: Iterable[float],
    dimensions: Iterable[float],
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    *,
    bevel: float = 0.0,
    occluder: bool = False,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=tuple(location))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = tuple(dimensions)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0.0:
        modifier = obj.modifiers.new("Soft edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    obj["synthetic_occluder"] = bool(occluder)
    _move_to_collection(obj, collection)
    return obj


def _add_surround_ring(material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    """Create a coplanar free-zone ring without overlapping the court top."""

    bounds = (
        (-9.0, 0.0, -9.0, 27.0),
        (COURT_WIDTH, 18.0, -9.0, 27.0),
        (0.0, COURT_WIDTH, -9.0, 0.0),
        (0.0, COURT_WIDTH, COURT_LENGTH, 27.0),
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for x0, x1, y0, y1 in bounds:
        start = len(vertices)
        vertices.extend(((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)))
        faces.append((start, start + 1, start + 2, start + 3))
    mesh = bpy.data.meshes.new("COURT_surround_ring_data")
    mesh.from_pydata(vertices, (), faces)
    mesh.update()
    obj = bpy.data.objects.new("COURT_surround_ring", mesh)
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def _add_cylinder_between(
    name: str,
    start: Iterable[float],
    end: Iterable[float],
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    *,
    vertices: int = 12,
    occluder: bool = False,
) -> bpy.types.Object:
    start_v = Vector(tuple(start))
    end_v = Vector(tuple(end))
    direction = end_v - start_v
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=direction.length,
        location=(start_v + end_v) * 0.5,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.data.materials.append(material)
    obj["synthetic_occluder"] = bool(occluder)
    _move_to_collection(obj, collection)
    return obj


def _add_uv_sphere(
    name: str,
    location: Iterable[float],
    radius: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    *,
    occluder: bool = False,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=tuple(location))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    obj["synthetic_occluder"] = bool(occluder)
    _move_to_collection(obj, collection)
    return obj


def _parent_local(obj: bpy.types.Object, root: bpy.types.Object) -> bpy.types.Object:
    obj.parent = root
    return obj


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    for child in tuple(scene.collection.children):
        bpy.data.collections.remove(child)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in tuple(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def _build_net(static: bpy.types.Collection, materials: dict[str, bpy.types.Material], net_height: float) -> None:
    post_x = (-0.62, COURT_WIDTH + 0.62)
    for side, x in enumerate(post_x):
        _add_cylinder_between(
            f"NET_post_{side}",
            (x, 9.0, 0.0),
            (x, 9.0, net_height + 0.42),
            0.075,
            materials["post"],
            static,
            vertices=20,
            occluder=True,
        )
        _add_cylinder_between(
            f"NET_antenna_{side}",
            (0.0 if side == 0 else COURT_WIDTH, 9.0, net_height - 0.80),
            (0.0 if side == 0 else COURT_WIDTH, 9.0, net_height + 0.80),
            0.012,
            materials["antenna"],
            static,
            vertices=10,
            occluder=True,
        )

    bottom = net_height - 1.0
    mesh = bpy.data.meshes.new("NET_texture_plane_data")
    mesh.from_pydata(
        (
            (0.0, 9.0, bottom),
            (COURT_WIDTH, 9.0, bottom),
            (COURT_WIDTH, 9.0, net_height),
            (0.0, 9.0, net_height),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_by_vertex = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            uv_layer.data[loop_index].uv = uv_by_vertex[mesh.loops[loop_index].vertex_index]
    net = bpy.data.objects.new("NET_texture_plane", mesh)
    net.data.materials.append(materials["net_texture"])
    # The alpha plane is visual only. Visibility uses the actual grid spacing,
    # so transparent holes are not incorrectly marked as occluded.
    net["synthetic_ray_ignore"] = True
    static.objects.link(net)
    _add_box("NET_top_tape", (4.5, 9.0, net_height), (9.15, 0.035, 0.07), materials["line"], static, occluder=True)
    _add_box("NET_bottom_tape", (4.5, 9.0, bottom), (9.10, 0.025, 0.04), materials["line"], static, occluder=True)


def _build_mannequin(
    index: int,
    dynamic: bpy.types.Collection,
    materials: dict[str, bpy.types.Material],
) -> bpy.types.Object:
    root = bpy.data.objects.new(f"PLAYER_{index:02d}", None)
    root.empty_display_type = "PLAIN_AXES"
    root["synthetic_player"] = True
    dynamic.objects.link(root)
    jersey = materials["team_a"] if index % 2 == 0 else materials["team_b"]

    head = _add_uv_sphere(f"PLAYER_{index:02d}_head", (0.0, 0.0, 1.73), 0.125, materials["skin"], dynamic, occluder=True)
    torso = _add_box(f"PLAYER_{index:02d}_torso", (0.0, 0.0, 1.30), (0.43, 0.25, 0.62), jersey, dynamic, bevel=0.10, occluder=True)
    hips = _add_box(f"PLAYER_{index:02d}_shorts", (0.0, 0.0, 0.92), (0.34, 0.24, 0.24), materials["shorts"], dynamic, bevel=0.06, occluder=True)
    left_leg = _add_cylinder_between(f"PLAYER_{index:02d}_leg_l", (-0.10, 0.0, 0.86), (-0.11, 0.0, 0.08), 0.065, materials["skin"], dynamic, occluder=True)
    right_leg = _add_cylinder_between(f"PLAYER_{index:02d}_leg_r", (0.10, 0.0, 0.86), (0.11, 0.0, 0.08), 0.065, materials["skin"], dynamic, occluder=True)
    arm_angle = 0.25 if index % 3 else 0.65
    left_arm = _add_cylinder_between(f"PLAYER_{index:02d}_arm_l", (-0.21, 0.0, 1.48), (-0.38, 0.02, 1.48 + arm_angle), 0.055, materials["skin"], dynamic, occluder=True)
    right_arm = _add_cylinder_between(f"PLAYER_{index:02d}_arm_r", (0.21, 0.0, 1.48), (0.38, -0.02, 1.48 + arm_angle), 0.055, materials["skin"], dynamic, occluder=True)
    for part in (head, torso, hips, left_leg, right_leg, left_arm, right_arm):
        _parent_local(part, root)
    return root


def _build_arena_props(
    dynamic: bpy.types.Collection,
    materials: dict[str, bpy.types.Material],
    pool_size: int,
) -> list[bpy.types.Object]:
    props: list[bpy.types.Object] = []
    for index in range(pool_size):
        prop = _add_box(
            f"DISTRACTION_{index:02d}",
            (0.0, 0.0, 0.5),
            (1.4, 0.18, 1.0),
            materials["board_a"] if index % 2 == 0 else materials["board_b"],
            dynamic,
            bevel=0.04,
            occluder=True,
        )
        prop["synthetic_distraction"] = True
        props.append(prop)
    return props


def _tag_arena_object(obj: bpy.types.Object, profile: str, side: int = 0) -> bpy.types.Object:
    obj["arena_profile"] = profile
    obj["arena_side"] = int(side)
    return obj


def _build_crowd_member(
    index: int,
    location: tuple[float, float, float],
    *,
    profile: str,
    side: int,
    collection: bpy.types.Collection,
    materials: dict[str, bpy.types.Material],
) -> bpy.types.Object:
    root = bpy.data.objects.new(f"AUDIENCE_{profile}_{index:03d}", None)
    collection.objects.link(root)
    root["arena_profile"] = profile
    root["arena_side"] = int(side)

    shirt_materials = (
        materials["crowd_red"],
        materials["crowd_blue"],
        materials["crowd_white"],
        materials["crowd_dark"],
    )
    torso = _add_box(
        f"AUDIENCE_TORSO_{profile}_{index:03d}",
        (0.0, 0.0, 0.62),
        (0.34, 0.26, 0.64),
        shirt_materials[index % len(shirt_materials)],
        collection,
        bevel=0.035,
    )
    torso.parent = root
    torso["audience_torso"] = True

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.18, location=(0.0, 0.0, 1.10))
    head = bpy.context.object
    head.name = f"AUDIENCE_HEAD_{profile}_{index:03d}"
    head.data.materials.append(materials["skin"])
    head.parent = root
    _move_to_collection(head, collection)

    root.location = location
    root.rotation_euler = (0.0, 0.0, math.radians(90.0 if side < 0 else -90.0))
    return root


def _build_arena_backgrounds(
    collection: bpy.types.Collection,
    materials: dict[str, bpy.types.Material],
) -> tuple[list[bpy.types.Object], list[bpy.types.Object], list[bpy.types.Object]]:
    """Build switchable championship and small-gym background variants."""

    backgrounds: list[bpy.types.Object] = []
    audience: list[bpy.types.Object] = []
    led_boards: list[bpy.types.Object] = []

    def add_background(
        profile: str,
        name: str,
        location: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        material: bpy.types.Material,
        *,
        side: int = 0,
        bevel: float = 0.0,
    ) -> bpy.types.Object:
        obj = _add_box(name, location, dimensions, material, collection, bevel=bevel)
        backgrounds.append(_tag_arena_object(obj, profile, side))
        return obj

    # Large international-event arena: red tiered seating and bright LED
    # ribbons on both sidelines, matching the dense background of broadcasts.
    for side in (-1, 1):
        wall_x = -13.0 if side < 0 else 22.0
        add_background(
            "championship",
            f"ARENA_CHAMP_WALL_{side:+d}",
            (wall_x, 9.0, 5.5),
            (0.35, 44.0, 11.0),
            materials["arena_wall_dark"],
            side=side,
        )
        inner_x = -5.1 if side < 0 else 14.1
        for tier in range(5):
            x = inner_x + side * tier * 1.15
            z = 0.42 + tier * 0.78
            add_background(
                "championship",
                f"ARENA_CHAMP_TIER_{side:+d}_{tier}",
                (x, 9.0, z),
                (1.15, 30.0, 0.72),
                materials["stand_red"],
                side=side,
                bevel=0.025,
            )
            for seat in range(9):
                y = -1.2 + seat * 2.55 + (tier % 2) * 0.35
                audience.append(
                    _build_crowd_member(
                        len(audience),
                        (x - side * 0.08, y, z + 0.38),
                        profile="championship",
                        side=side,
                        collection=collection,
                        materials=materials,
                    )
                )
        for segment in range(8):
            y = -0.8 + segment * 2.75
            led = add_background(
                "championship",
                f"LED_CHAMP_{side:+d}_{segment:02d}",
                (-3.75 if side < 0 else 12.75, y, 0.58),
                (0.18, 2.55, 0.88),
                materials["led_atlas"][segment % len(materials["led_atlas"])],
                side=side,
                bevel=0.025,
            )
            led["arena_led"] = True
            led_boards.append(led)

    for end_side, y in ((-1, -3.75), (1, 21.75)):
        for segment in range(4):
            x = 0.25 + segment * 2.82
            led = add_background(
                "championship",
                f"LED_CHAMP_END_{end_side:+d}_{segment:02d}",
                (x, y, 0.58),
                (2.55, 0.18, 0.88),
                materials["led_atlas"][(segment + 3) % len(materials["led_atlas"])],
                side=0,
                bevel=0.025,
            )
            led["arena_led"] = True
            led_boards.append(led)

    # Bright international hall with yellow double-sided grandstands, sparse
    # spectators, a high truss roof and a central media/scoring box.
    for side in (-1, 1):
        wall_x = -14.0 if side < 0 else 23.0
        add_background(
            "yellow_grandstand",
            f"ARENA_YELLOW_WALL_{side:+d}",
            (wall_x, 9.0, 6.4),
            (0.35, 46.0, 12.8),
            materials["arena_wall_light"],
            side=side,
        )
        inner_x = -5.2 if side < 0 else 14.2
        for tier in range(5):
            x = inner_x + side * tier * 1.18
            z = 0.40 + tier * 0.80
            add_background(
                "yellow_grandstand",
                f"ARENA_YELLOW_TIER_{side:+d}_{tier}",
                (x, 9.0, z),
                (1.18, 31.0, 0.70),
                materials["stand_yellow"],
                side=side,
                bevel=0.025,
            )
            for seat in range(8):
                y = -0.4 + seat * 2.75 + (tier % 2) * 0.30
                audience.append(
                    _build_crowd_member(
                        len(audience),
                        (x - side * 0.08, y, z + 0.36),
                        profile="yellow_grandstand",
                        side=side,
                        collection=collection,
                        materials=materials,
                    )
                )
        media_box = add_background(
            "yellow_grandstand",
            f"ARENA_YELLOW_MEDIA_{side:+d}",
            (wall_x - side * 0.22, 9.0, 4.3),
            (0.18, 5.2, 2.3),
            materials["media_white"],
            side=side,
            bevel=0.04,
        )
        media_box["arena_media_box"] = True
        for segment in range(8):
            y = -0.8 + segment * 2.75
            led = add_background(
                "yellow_grandstand",
                f"LED_YELLOW_{side:+d}_{segment:02d}",
                (-3.75 if side < 0 else 12.75, y, 0.58),
                (0.18, 2.55, 0.88),
                materials["led_atlas"][(segment + 2) % len(materials["led_atlas"])],
                side=side,
                bevel=0.025,
            )
            led["arena_led"] = True
            led_boards.append(led)

    for end_side, y in ((-1, -3.75), (1, 21.75)):
        for segment in range(4):
            x = 0.25 + segment * 2.82
            led = add_background(
                "yellow_grandstand",
                f"LED_YELLOW_END_{end_side:+d}_{segment:02d}",
                (x, y, 0.58),
                (2.55, 0.18, 0.88),
                materials["led_atlas"][(segment + 1) % len(materials["led_atlas"])],
                side=0,
                bevel=0.025,
            )
            led["arena_led"] = True
            led_boards.append(led)

    for beam_index, y in enumerate((-6.0, 1.5, 9.0, 16.5, 24.0)):
        add_background(
            "yellow_grandstand",
            f"ARENA_YELLOW_ROOF_BEAM_{beam_index}",
            (4.5, y, 12.2),
            (35.0, 0.30, 0.30),
            materials["beam_dark"],
            side=0,
        )

    # Small steel-roof gym: pale sheet-metal walls, visible posts/trusses,
    # one-sided bleachers and large generic event banners.
    for side in (-1, 1):
        wall_x = -11.0 if side < 0 else 20.0
        add_background(
            "small_gym",
            f"ARENA_GYM_WALL_{side:+d}",
            (wall_x, 9.0, 5.8),
            (0.30, 42.0, 11.6),
            materials["arena_wall_light"],
            side=0,
        )
        for post_index, y in enumerate((-7.0, 1.0, 9.0, 17.0, 25.0)):
            add_background(
                "small_gym",
                f"ARENA_GYM_POST_{side:+d}_{post_index}",
                (wall_x - side * 0.22, y, 5.8),
                (0.32, 0.32, 11.6),
                materials["beam_dark"],
                side=0,
            )
        banner = add_background(
            "small_gym",
            f"ARENA_GYM_BANNER_{side:+d}",
            (wall_x - side * 0.20, 9.0, 7.0),
            (0.12, 11.0, 3.4),
            materials["banner_navy"] if side < 0 else materials["banner_red"],
            side=side,
            bevel=0.02,
        )
        banner["arena_banner"] = True

        inner_x = -4.8 if side < 0 else 13.8
        for tier in range(4):
            x = inner_x + side * tier * 1.0
            z = 0.38 + tier * 0.72
            add_background(
                "small_gym",
                f"ARENA_GYM_TIER_{side:+d}_{tier}",
                (x, 11.0, z),
                (1.0, 15.5, 0.64),
                materials["stand_red"],
                side=side,
                bevel=0.02,
            )
            for seat in range(7):
                y = 3.4 + seat * 2.1 + (tier % 2) * 0.25
                audience.append(
                    _build_crowd_member(
                        len(audience),
                        (x - side * 0.08, y, z + 0.34),
                        profile="small_gym",
                        side=side,
                        collection=collection,
                        materials=materials,
                    )
                )

    for beam_index, y in enumerate((-5.0, 2.0, 9.0, 16.0, 23.0)):
        add_background(
            "small_gym",
            f"ARENA_GYM_ROOF_BEAM_{beam_index}",
            (4.5, y, 11.0),
            (31.0, 0.28, 0.28),
            materials["beam_dark"],
            side=0,
        )

    return backgrounds, audience, led_boards


def _add_area_light(name: str, location: tuple[float, float, float], energy: float, collection: bpy.types.Collection) -> bpy.types.Object:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = 7.0
    if hasattr(data, "specular_factor"):
        data.specular_factor = 0.0
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    collection.objects.link(obj)
    _look_at(obj, COURT_CENTER + Vector((0.0, 0.0, 0.4)))
    return obj


def _look_at(obj: bpy.types.Object, target: Vector, roll: float = 0.0) -> None:
    direction = target - obj.location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    if roll:
        obj.rotation_quaternion = obj.rotation_quaternion @ Quaternion((0.0, 0.0, 1.0), roll)


def build_scene(
    seed: int = 20260811,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the reusable arena, court, net, players, lights and camera."""

    config = config or load_randomization_config()
    rng = random.Random(seed)
    _clear_scene()
    scene = bpy.context.scene
    scene.name = "Synthetic Volleyball Court 36"
    scene["court_schema"] = "canonical-36-v1"
    scene["coordinate_convention"] = "x=width-left-to-right,y=length-far-to-near,z=up-metres"

    static = _collection("STATIC_COURT")
    dynamic = _collection("DYNAMIC_OCCLUDERS")
    arena_background = _collection("ARENA_BACKGROUNDS")
    lights = _collection("LIGHTS")
    cameras = _collection("CAMERAS")

    palette_entries = config["materials"]["palettes"]
    palette = next(iter(palette_entries.values()))
    materials = {
        "court": _court_surface_material(palette["court"], (1.0, 1.0, 1.0, 1.0)),
        "surround": _surround_surface_material(palette["surround"], (1.0, 1.0, 1.0, 1.0)),
        "arena": _noisy_material("MAT_Arena", (0.055, 0.06, 0.075, 1.0), roughness=0.32, noise_scale=3.0, noise_strength=0.22),
        "line": _simple_material("MAT_Lines", (1.0, 1.0, 1.0, 1.0), roughness=0.12),
        "post": _simple_material("MAT_Post", (0.06, 0.10, 0.22, 1.0), roughness=0.22, metallic=0.25),
        "antenna": _simple_material("MAT_Antenna", (0.95, 0.06, 0.04, 1.0), roughness=0.30),
        "net_texture": _net_texture_material(),
        "skin": _simple_material("MAT_Skin", (0.58, 0.28, 0.16, 1.0), roughness=0.54),
        "team_a": _simple_material("MAT_Team_A", palette["team_a"], roughness=0.42),
        "team_b": _simple_material("MAT_Team_B", palette["team_b"], roughness=0.42),
        "shorts": _simple_material("MAT_Shorts", (0.015, 0.02, 0.03, 1.0), roughness=0.48),
        "board_a": _simple_material("MAT_Board_A", (0.04, 0.15, 0.52, 1.0), roughness=0.30),
        "board_b": _simple_material("MAT_Board_B", (0.65, 0.03, 0.08, 1.0), roughness=0.30),
        "arena_wall_dark": _simple_material("MAT_ArenaWallDark", (0.035, 0.025, 0.035, 1.0), roughness=0.62),
        "arena_wall_light": _simple_material("MAT_ArenaWallLight", (0.46, 0.47, 0.46, 1.0), roughness=0.66),
        "stand_red": _simple_material("MAT_StandRed", (0.24, 0.018, 0.025, 1.0), roughness=0.58),
        "stand_yellow": _simple_material("MAT_StandYellow", (0.58, 0.39, 0.035, 1.0), roughness=0.58),
        "media_white": _simple_material("MAT_MediaWhite", (0.70, 0.72, 0.72, 1.0), roughness=0.55),
        "beam_dark": _simple_material("MAT_BeamDark", (0.025, 0.028, 0.032, 1.0), roughness=0.42, metallic=0.35),
        "banner_navy": _simple_material("MAT_BannerNavy", (0.035, 0.055, 0.28, 1.0), roughness=0.48),
        "banner_red": _simple_material("MAT_BannerRed", (0.48, 0.018, 0.035, 1.0), roughness=0.48),
        "led_blue": _emissive_material("MAT_LED_Blue", (0.025, 0.12, 0.95, 1.0), strength=2.6),
        "led_white": _emissive_material("MAT_LED_White", (0.92, 0.96, 1.0, 1.0), strength=2.1),
        "led_red": _emissive_material("MAT_LED_Red", (0.95, 0.025, 0.055, 1.0), strength=2.5),
        "crowd_red": _simple_material("MAT_CrowdRed", (0.55, 0.025, 0.035, 1.0), roughness=0.56),
        "crowd_blue": _simple_material("MAT_CrowdBlue", (0.025, 0.08, 0.42, 1.0), roughness=0.56),
        "crowd_white": _simple_material("MAT_CrowdWhite", (0.72, 0.74, 0.76, 1.0), roughness=0.56),
        "crowd_dark": _simple_material("MAT_CrowdDark", (0.025, 0.03, 0.04, 1.0), roughness=0.56),
        "led_atlas": _advertising_atlas_materials(),
    }

    _add_box("ARENA_floor", (4.5, 9.0, -0.15), (46.0, 56.0, 0.20), materials["arena"], static)
    _add_surround_ring(materials["surround"], static)
    _add_box("COURT_surface", (4.5, 9.0, -0.03), (9.0, 18.0, 0.06), materials["court"], static)

    _, net_height_entry = _weighted_entry(
        rng,
        config,
        config["scene"]["net_heights"],
        "scene.net_heights",
    )
    net_height = float(net_height_entry["value_m"])
    scene["net_height"] = float(net_height)
    _build_net(static, materials, net_height)

    # Referee stand and side equipment make the arena less sterile and create
    # realistic partial court occlusions near the net posts.
    _add_box("PROP_referee_stand", (10.05, 9.0, 0.85), (0.72, 1.10, 1.70), materials["post"], static, bevel=0.04, occluder=True)
    _add_box("PROP_score_table", (-3.4, 9.0, 0.45), (2.8, 0.75, 0.90), materials["board_a"], static, bevel=0.05, occluder=True)

    player_pool_size = int(config["scene"]["player_pool_size"])
    distraction_pool_size = int(config["scene"]["distraction_pool_size"])
    if player_pool_size < 32:
        raise ValueError("scene.player_pool_size must be at least 32")
    if distraction_pool_size < 1:
        raise ValueError("scene.distraction_pool_size must be positive")
    players = [_build_mannequin(index, dynamic, materials) for index in range(player_pool_size)]
    props = _build_arena_props(dynamic, materials, distraction_pool_size)
    arena_backgrounds, audience, led_boards = _build_arena_backgrounds(arena_background, materials)

    camera_data = bpy.data.cameras.new("SyntheticCamera")
    camera_data.type = "PERSP"
    camera_data.lens = 45.0
    camera_data.sensor_width = 36.0
    camera = bpy.data.objects.new("SyntheticCamera", camera_data)
    cameras.objects.link(camera)
    scene.camera = camera

    sun_data = bpy.data.lights.new("LIGHT_sun", "SUN")
    sun_data.energy = 1.2
    if hasattr(sun_data, "specular_factor"):
        sun_data.specular_factor = 0.10
    sun = bpy.data.objects.new("LIGHT_sun", sun_data)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(32.0))
    lights.objects.link(sun)
    for index, location in enumerate(((-6.0, -5.0, 14.0), (15.0, -5.0, 15.0), (-6.0, 23.0, 15.0), (15.0, 23.0, 14.0))):
        _add_area_light(f"LIGHT_area_{index}", location, 1400.0, lights)

    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.025, 0.035, 0.06, 1.0)
    background.inputs["Strength"].default_value = 0.52

    return {
        "scene": scene,
        "camera": camera,
        "players": players,
        "props": props,
        "arena_backgrounds": arena_backgrounds,
        "audience": audience,
        "led_boards": led_boards,
        "materials": materials,
        "net_height": net_height,
        "config": config,
    }


def configure_render(
    scene: bpy.types.Scene,
    *,
    config: dict[str, Any] | None = None,
    width: int = 1280,
    height: int = 720,
    samples: int = 48,
    engine: str = "BLENDER_EEVEE",
) -> None:
    requested_engine = engine
    if engine == "BLENDER_EEVEE_NEXT":
        engine = "BLENDER_EEVEE"
    try:
        scene.render.engine = engine
    except TypeError:
        if requested_engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
            engine = "BLENDER_EEVEE_NEXT"
        else:
            raise
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 94
    scene.render.use_file_extension = True
    scene.render.film_transparent = False
    if engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        scene.render.image_settings.color_depth = "8"
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = samples
        if config is not None and hasattr(scene, "eevee") and hasattr(scene.eevee, "use_raytracing"):
            lighting_config = config["lighting"]
            scene.eevee.use_raytracing = bool(lighting_config["enable_eevee_raytracing"])
            if hasattr(scene.eevee, "ray_tracing_method"):
                scene.eevee.ray_tracing_method = str(lighting_config["eevee_raytracing_method"])
    elif engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    scene.view_settings.look = "AgX - Medium High Contrast"


def _sample_capture_profile(
    rng: random.Random,
    *,
    config: dict[str, Any],
    base_width: int,
    base_height: int,
    minimum_height: int,
    enabled: bool,
) -> dict[str, Any]:
    """Choose realistic broadcast/video degradation without changing geometry."""

    if not enabled:
        return {
            "resolution": [base_width, base_height],
            "jpeg_quality": 94,
            "filter": "clean",
            "white_balance_enabled": False,
            "white_balance_temperature": 6500.0,
            "white_balance_tint": 10.0,
            "exposure_offset": 0.0,
            "gamma": 1.0,
            "channel_gain": [1.0, 1.0, 1.0],
            "sensor_noise_sigma": 0.0,
            "blur_mode": "none",
            "blur_radius_px": 0,
            "blur_angle_degrees": 0.0,
            "chromatic_shift_px": 0,
            "vignette_strength": 0.0,
            "fake_ui_enabled": False,
            "fake_ui_bbox": [0.0, 0.0, 0.0, 0.0],
            "fake_ui_score": {"a_sets": 0, "b_sets": 0, "a_points": 0, "b_points": 0},
        }

    capture_config = config["capture"]
    minimum_height = min(base_height, max(1, int(minimum_height)))
    resolution_entries = {
        name: entry
        for name, entry in capture_config["resolution_heights"].items()
        if minimum_height <= int(entry["value"]) <= base_height
    }
    existing_values = {int(entry["value"]) for entry in resolution_entries.values()}
    for endpoint_name, endpoint_value in (("minimum", minimum_height), ("maximum", base_height)):
        if endpoint_value not in existing_values:
            resolution_entries[endpoint_name] = {
                "stars": "★★★☆☆",
                "weight": 1,
                "value": endpoint_value,
            }
    _, resolution_entry = _weighted_entry(
        rng,
        config,
        resolution_entries,
        "capture.resolution_heights",
    )
    target_height = int(resolution_entry["value"])
    target_width = max(2, int(round(base_width * target_height / base_height / 2.0)) * 2)

    filter_name, _ = _weighted_entry(rng, config, capture_config["filters"], "capture.filters")
    channel_gain = [
        _uniform(rng, capture_config["base_channel_gain"], "capture.base_channel_gain")
        for _ in range(3)
    ]
    exposure_offset = _uniform(rng, capture_config["exposure_offset"], "capture.exposure_offset")
    gamma = _uniform(rng, capture_config["gamma"], "capture.gamma")
    if filter_name == "warm_broadcast":
        channel_gain[0] *= _uniform(rng, capture_config["warm_red_gain"], "capture.warm_red_gain")
        channel_gain[2] *= _uniform(rng, capture_config["warm_blue_gain"], "capture.warm_blue_gain")
    elif filter_name == "cool_broadcast":
        channel_gain[0] *= _uniform(rng, capture_config["cool_red_gain"], "capture.cool_red_gain")
        channel_gain[2] *= _uniform(rng, capture_config["cool_blue_gain"], "capture.cool_blue_gain")
    elif filter_name == "green_cast":
        channel_gain[1] *= _uniform(rng, capture_config["green_gain"], "capture.green_gain")
    elif filter_name == "magenta_cast":
        channel_gain[0] *= _uniform(rng, capture_config["magenta_red_gain"], "capture.magenta_red_gain")
        channel_gain[1] *= _uniform(rng, capture_config["magenta_green_gain"], "capture.magenta_green_gain")
        channel_gain[2] *= _uniform(rng, capture_config["magenta_blue_gain"], "capture.magenta_blue_gain")
    elif filter_name == "washed_broadcast":
        exposure_offset += _uniform(rng, capture_config["washed_exposure_add"], "capture.washed_exposure_add")
        gamma *= _uniform(rng, capture_config["washed_gamma_multiplier"], "capture.washed_gamma_multiplier")

    blur_mode, _ = _weighted_entry(rng, config, capture_config["blur_modes"], "capture.blur_modes")
    if blur_mode == "none":
        blur_radius = 0
    elif blur_mode == "motion":
        blur_radius = int(rng.choice(capture_config["motion_blur_radii"]))
    elif blur_mode == "whip_pan":
        reference_radius = int(rng.choice(capture_config["whip_pan_reference_radii_720p"]))
        blur_radius = max(3, round(reference_radius * target_height / 720.0))
    else:
        blur_radius = int(rng.choice(capture_config["defocus_radii"]))
    resolution_factor = math.sqrt(base_height / target_height)
    noise_sigma = _uniform(rng, capture_config["sensor_noise_sigma"], "capture.sensor_noise_sigma") * resolution_factor
    if rng.random() < float(capture_config["strong_noise_probability"]):
        noise_sigma *= _uniform(rng, capture_config["strong_noise_multiplier"], "capture.strong_noise_multiplier")

    fake_ui_config = capture_config["fake_ui"]
    fake_ui_enabled = rng.random() < float(fake_ui_config["enabled_probability"])
    ui_width = _uniform(rng, fake_ui_config["width"], "capture.fake_ui.width")
    ui_height = ui_width * _uniform(rng, fake_ui_config["height_ratio"], "capture.fake_ui.height_ratio")
    ui_margin = _uniform(rng, fake_ui_config["horizontal_margin"], "capture.fake_ui.horizontal_margin")
    ui_x = ui_margin if rng.random() < float(fake_ui_config["left_probability"]) else 1.0 - ui_width - ui_margin
    ui_y = _uniform(rng, fake_ui_config["top"], "capture.fake_ui.top")
    _, chromatic_entry = _weighted_entry(
        rng,
        config,
        capture_config["chromatic_shift"],
        "capture.chromatic_shift",
    )

    return {
        "resolution": [target_width, target_height],
        "jpeg_quality": _randint(rng, capture_config["jpeg_quality"], "capture.jpeg_quality"),
        "filter": filter_name,
        "white_balance_enabled": True,
        "white_balance_temperature": _uniform(rng, capture_config["white_balance_temperature"], "capture.white_balance_temperature"),
        "white_balance_tint": _uniform(rng, capture_config["white_balance_tint"], "capture.white_balance_tint"),
        "exposure_offset": exposure_offset,
        "gamma": gamma,
        "channel_gain": channel_gain,
        "sensor_noise_sigma": noise_sigma,
        "blur_mode": blur_mode,
        "blur_radius_px": blur_radius,
        "blur_angle_degrees": (
            _uniform(rng, capture_config["whip_pan_angle_degrees"], "capture.whip_pan_angle_degrees")
            if blur_mode == "whip_pan"
            else _uniform(rng, capture_config["motion_blur_angle_degrees"], "capture.motion_blur_angle_degrees")
            if blur_mode == "motion"
            else 0.0
        ),
        "chromatic_shift_px": int(chromatic_entry["pixels"]),
        "vignette_strength": _uniform(rng, capture_config["vignette_strength"], "capture.vignette_strength"),
        "fake_ui_enabled": fake_ui_enabled,
        "fake_ui_bbox": [ui_x, ui_y, ui_width, ui_height] if fake_ui_enabled else [0.0, 0.0, 0.0, 0.0],
        "fake_ui_score": {
            "a_sets": _randint(rng, fake_ui_config["sets"], "capture.fake_ui.sets"),
            "b_sets": _randint(rng, fake_ui_config["sets"], "capture.fake_ui.sets"),
            "a_points": _randint(rng, fake_ui_config["points"], "capture.fake_ui.points"),
            "b_points": _randint(rng, fake_ui_config["points"], "capture.fake_ui.points"),
        },
    }


def _apply_capture_profile(scene: bpy.types.Scene, profile: dict[str, Any]) -> None:
    width, height = profile["resolution"]
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    # Render a clean intermediate. The requested JPEG quality is applied after
    # camera artifacts so compression is the final step in the pipeline.
    scene.render.image_settings.quality = 100
    scene.render.use_motion_blur = False
    scene.view_settings.use_white_balance = bool(profile["white_balance_enabled"])
    scene.view_settings.white_balance_temperature = float(profile["white_balance_temperature"])
    scene.view_settings.white_balance_tint = float(profile["white_balance_tint"])
    scene.view_settings.gamma = float(profile["gamma"])


def _shift_image_edge(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return image
    height, width = image.shape[:2]
    pad = max(1, abs(dx), abs(dy))
    if image.ndim == 2:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
    else:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return padded[pad - dy : pad - dy + height, pad - dx : pad - dx + width]


def _blur_rgb(rgb: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    mode = str(profile["blur_mode"])
    radius = int(profile["blur_radius_px"])
    if mode == "none" or radius <= 0:
        return rgb
    if mode == "defocus":
        offsets = tuple((dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1))
    else:
        angle = math.radians(float(profile["blur_angle_degrees"]))
        offsets = tuple(
            (round(math.cos(angle) * step), round(math.sin(angle) * step))
            for step in range(-radius, radius + 1)
        )
    blurred = np.zeros_like(rgb)
    for dx, dy in offsets:
        blurred += _shift_image_edge(rgb, int(dx), int(dy))
    return blurred / float(len(offsets))


def _fill_canvas_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[float, float, float],
) -> None:
    height, width = canvas.shape[:2]
    left, right = max(0, x0), min(width, x1)
    top, bottom = max(0, y0), min(height, y1)
    if left < right and top < bottom:
        canvas[top:bottom, left:right, :] = color


def _draw_seven_segment_digit(
    canvas: np.ndarray,
    digit: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[float, float, float],
) -> None:
    segment_map = {
        0: "abcedf",
        1: "bc",
        2: "abdeg",
        3: "abcdg",
        4: "bcfg",
        5: "acdfg",
        6: "acdefg",
        7: "abc",
        8: "abcdefg",
        9: "abcdfg",
    }
    active = set(segment_map[int(digit) % 10])
    thickness = max(1, min(width, height) // 7)
    middle = y + height // 2
    segments = {
        "a": (x + thickness, y, x + width - thickness, y + thickness),
        "b": (x + width - thickness, y + thickness, x + width, middle),
        "c": (x + width - thickness, middle, x + width, y + height - thickness),
        "d": (x + thickness, y + height - thickness, x + width - thickness, y + height),
        "e": (x, middle, x + thickness, y + height - thickness),
        "f": (x, y + thickness, x + thickness, middle),
        "g": (x + thickness, middle - thickness // 2, x + width - thickness, middle + (thickness + 1) // 2),
    }
    for name in active:
        _fill_canvas_rect(canvas, *segments[name], color)


def _draw_bitmap_letter(
    canvas: np.ndarray,
    letter: str,
    x: int,
    y: int,
    cell: int,
    color: tuple[float, float, float],
) -> None:
    font = {
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    }
    for row, bits in enumerate(font[letter]):
        for column, bit in enumerate(bits):
            if bit == "1":
                _fill_canvas_rect(
                    canvas,
                    x + column * cell,
                    y + row * cell,
                    x + (column + 1) * cell,
                    y + (row + 1) * cell,
                    color,
                )


def _draw_fake_broadcast_ui(rgb: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    if not profile.get("fake_ui_enabled"):
        return rgb
    height, width = rgb.shape[:2]
    x_norm, y_norm, width_norm, height_norm = (float(value) for value in profile["fake_ui_bbox"])
    ui_width = max(120, round(width * width_norm))
    ui_height = max(52, round(height * height_norm))
    left = max(0, min(width - ui_width, round(width * x_norm)))
    top = max(0, min(height - ui_height, round(height * y_norm)))
    bottom_index = height - top - ui_height

    canvas = np.empty((ui_height, ui_width, 3), dtype=np.float32)
    canvas[:, :, :] = (0.43, 0.43, 0.46)
    alpha = np.full((ui_height, ui_width, 1), 0.94, dtype=np.float32)
    corner = max(4, ui_height // 10)
    alpha[:corner, :corner, :] = 0.0
    alpha[:corner, -corner:, :] = 0.0
    alpha[-corner:, :corner, :] = 0.0
    alpha[-corner:, -corner:, :] = 0.0

    label_right = round(ui_width * 0.55)
    score_right = round(ui_width * 0.75)
    row_height = ui_height // 2
    canvas[:, :label_right, :] = (0.84, 0.85, 0.88)
    canvas[:, label_right:score_right, :] = (0.055, 0.06, 0.085)
    canvas[:, score_right:, :] = (0.075, 0.08, 0.105)
    _fill_canvas_rect(canvas, 0, row_height - 1, ui_width, row_height + 1, (0.30, 0.31, 0.34))
    _fill_canvas_rect(canvas, score_right - 1, 0, score_right + 1, ui_height, (0.34, 0.35, 0.38))

    arrow_width = max(6, ui_width // 22)
    for row_index, color in enumerate(((0.05, 0.16, 0.92), (0.92, 0.035, 0.045))):
        row_top = row_index * row_height
        center_y = row_top + row_height // 2
        for dx in range(arrow_width):
            half_height = max(1, round((dx + 1) / arrow_width * row_height * 0.22))
            _fill_canvas_rect(canvas, corner + dx, center_y - half_height, corner + dx + 1, center_y + half_height, color)
        cell = max(1, row_height // 10)
        _draw_bitmap_letter(
            canvas,
            "A" if row_index == 0 else "B",
            corner + arrow_width + cell * 3,
            row_top + max(2, (row_height - 7 * cell) // 2),
            cell,
            (0.025, 0.028, 0.04),
        )

    score = profile["fake_ui_score"]
    digit_height = max(12, round(row_height * 0.67))
    digit_width = max(7, round(digit_height * 0.52))
    for row_index, prefix in enumerate(("a", "b")):
        row_top = row_index * row_height
        digit_y = row_top + (row_height - digit_height) // 2
        set_x = label_right + max(3, (score_right - label_right - digit_width) // 2)
        _draw_seven_segment_digit(canvas, int(score[f"{prefix}_sets"]), set_x, digit_y, digit_width, digit_height, (0.95, 0.96, 1.0))
        points = int(score[f"{prefix}_points"])
        point_area_width = ui_width - score_right
        pair_width = digit_width * 2 + max(2, digit_width // 4)
        point_x = score_right + max(2, (point_area_width - pair_width) // 2)
        _draw_seven_segment_digit(canvas, points // 10, point_x, digit_y, digit_width, digit_height, (0.95, 0.96, 1.0))
        _draw_seven_segment_digit(
            canvas,
            points % 10,
            point_x + digit_width + max(2, digit_width // 4),
            digit_y,
            digit_width,
            digit_height,
            (0.95, 0.96, 1.0),
        )

    # Blender image pixels are bottom-up. Flip the top-down UI canvas before
    # compositing so the saved JPEG shows it at the requested screen corner.
    canvas = np.flipud(canvas)
    alpha = np.flipud(alpha)
    destination = rgb[bottom_index : bottom_index + ui_height, left : left + ui_width, :]
    shadow_top = max(0, bottom_index - max(2, ui_height // 18))
    shadow_left = min(width - ui_width, left + max(2, ui_height // 18))
    shadow = rgb[shadow_top : shadow_top + ui_height, shadow_left : shadow_left + ui_width, :]
    if shadow.shape == canvas.shape:
        shadow *= 0.82
    destination[:] = destination * (1.0 - alpha) + canvas * alpha
    return rgb


def _mark_broadcast_ui_occlusion(
    keypoints: list[dict[str, Any]],
    width: int,
    height: int,
    profile: dict[str, Any],
) -> int:
    if not profile.get("fake_ui_enabled"):
        return 0
    x_norm, y_norm, width_norm, height_norm = (float(value) for value in profile["fake_ui_bbox"])
    x0, x1 = x_norm * width, (x_norm + width_norm) * width
    y0, y1 = y_norm * height, (y_norm + height_norm) * height
    occluded = 0
    for row in keypoints:
        pixel = row.get("pixel")
        if pixel is None or int(row["visibility"]) != 2:
            continue
        if x0 <= float(pixel[0]) <= x1 and y0 <= float(pixel[1]) <= y1:
            row["visibility"] = 1
            row["occluded_by"] = "broadcast_ui"
            occluded += 1
    return occluded


def _save_degraded_render(image_path: Path, profile: dict[str, Any], sample_seed: int) -> None:
    """Read the rendered frame, add camera/video artifacts, and recompress it."""

    if not image_path.exists():
        raise RuntimeError(f"Blender did not write the intermediate render: {image_path}")
    rendered_image = bpy.data.images.load(str(image_path), check_existing=False)
    width, height = (int(value) for value in rendered_image.size)
    if width <= 0 or height <= 0:
        bpy.data.images.remove(rendered_image)
        raise RuntimeError(f"invalid rendered image dimensions {width}x{height}: {image_path}")
    pixels = np.empty(width * height * 4, dtype=np.float32)
    rendered_image.pixels.foreach_get(pixels)
    rgba = pixels.reshape((height, width, 4))
    rgb = rgba[:, :, :3].copy()

    rgb *= np.asarray(profile["channel_gain"], dtype=np.float32)[None, None, :]
    rgb = _blur_rgb(rgb, profile)

    chromatic_shift = int(profile["chromatic_shift_px"])
    if chromatic_shift:
        rgb[:, :, 0] = _shift_image_edge(rgb[:, :, 0], chromatic_shift, 0)
        rgb[:, :, 2] = _shift_image_edge(rgb[:, :, 2], -chromatic_shift, 0)

    vignette_strength = float(profile["vignette_strength"])
    if vignette_strength > 0.0:
        y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        vignette = 1.0 - vignette_strength * np.clip((x_axis * x_axis + y_axis * y_axis) * 0.5, 0.0, 1.0)
        rgb *= vignette[:, :, None]

    noise_sigma = float(profile["sensor_noise_sigma"])
    if noise_sigma > 0.0:
        noise_rng = np.random.default_rng(sample_seed ^ 0x51A7C0DE)
        luminance = np.maximum(0.15, np.sqrt(np.clip(rgb.mean(axis=2, keepdims=True), 0.0, None)))
        noise = noise_rng.normal(0.0, noise_sigma, rgb.shape).astype(np.float32)
        rgb += noise * luminance

    rgb = _draw_fake_broadcast_ui(rgb, profile)

    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, :, 3] = 1.0
    try:
        rendered_image.pixels.foreach_set(rgba.reshape(-1))
        rendered_image.update()
        rendered_image.save(
            filepath=str(image_path),
            quality=int(profile["jpeg_quality"]),
            save_copy=False,
        )
    finally:
        bpy.data.images.remove(rendered_image)


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


def _project_point(scene: bpy.types.Scene, camera: bpy.types.Object, point: Vector) -> tuple[float, float, float]:
    normalized = world_to_camera_view(scene, camera, point)
    width = float(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    height = float(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    return normalized.x * width, (1.0 - normalized.y) * height, normalized.z


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
    profile, _ = _weighted_entry(rng, config, arena_config["profiles"], "arena.profiles")
    if profile == "small_gym":
        active_side = (
            0
            if rng.random() < float(arena_config["small_gym_bilateral_probability"])
            else (-1 if rng.random() < float(arena_config["left_side_probability"]) else 1)
        )
    elif profile == "none":
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
    background.inputs["Strength"].default_value = _uniform(rng, lighting_config["world_strength"], "lighting.world_strength")
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
        **player_info,
        "distraction_count": distraction_count,
        **arena,
        **lighting,
    }


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
    fake_ui_count = 0
    rows: list[dict[str, Any]] = []
    for index in range(count):
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
        keypoints = _keypoint_labels(scene, camera)
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
            "schema": "court36-synthetic-blender-v1",
            "sample_index": index,
            "sample_seed": sample_seed,
            "split": split,
            "image": image_path.relative_to(output).as_posix(),
            "label": label_path.relative_to(output).as_posix(),
            "scene": scene_info,
            "capture": capture,
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
        "schema": "court36-synthetic-blender-v1",
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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-blend", type=Path)
    parser.add_argument("--resolution-x", type=int)
    parser.add_argument("--resolution-y", type=int)
    parser.add_argument("--minimum-resolution-y", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"),
        default=None,
    )
    parser.add_argument("--fixed-capture-quality", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)
    config = load_randomization_config(args.config)
    generation = config["generation"]
    count = int(generation["count"] if args.count is None else args.count)
    seed = int(generation["seed"] if args.seed is None else args.seed)
    resolution_x = int(generation["resolution_x"] if args.resolution_x is None else args.resolution_x)
    resolution_y = int(generation["resolution_y"] if args.resolution_y is None else args.resolution_y)
    minimum_resolution_y = int(
        generation["minimum_resolution_y"]
        if args.minimum_resolution_y is None
        else args.minimum_resolution_y
    )
    samples = int(generation["samples"] if args.samples is None else args.samples)
    engine = str(generation["engine"] if args.engine is None else args.engine)
    output = Path(generation["output"]) if args.output is None else args.output
    capture_randomization = bool(generation["capture_randomization"]) and not args.fixed_capture_quality
    generate_requested = args.save_blend is None or args.count is not None or args.output is not None
    if count < 0:
        raise ValueError("count cannot be negative")
    if resolution_x <= 0 or resolution_y <= 0:
        raise ValueError("render resolution must be positive")
    if minimum_resolution_y <= 0 or minimum_resolution_y > resolution_y:
        raise ValueError("minimum_resolution_y must be positive and cannot exceed resolution_y")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not count and args.save_blend is None:
        raise ValueError("set generation.count/--count or provide --save-blend")
    if set(config["splits"]) != {"train", "valid", "test"}:
        raise ValueError("splits must contain exactly train, valid, and test")
    state = build_scene(seed, config)
    configure_render(
        state["scene"],
        config=config,
        width=resolution_x,
        height=resolution_y,
        samples=samples,
        engine=engine,
    )
    randomize_scene(state, seed)
    if args.save_blend is not None:
        destination = args.save_blend.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(destination))
        print(json.dumps({"blend_file": str(destination), "objects": len(bpy.context.scene.objects)}, indent=2), flush=True)
    if generate_requested and count:
        manifest = generate_dataset(
            state,
            output,
            count=count,
            seed=seed,
            overwrite=args.overwrite,
            minimum_resolution_y=minimum_resolution_y,
            randomize_capture_quality=capture_randomization,
        )
        print(json.dumps({key: manifest[key] for key in ("count", "seed", "resolution", "split_counts", "visibility_counts")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
