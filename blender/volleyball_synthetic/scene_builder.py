"""Blender materials, court geometry, arena shells, and reusable scene construction."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from .config import _weighted_entry, load_randomization_config
from .constants import COURT_CENTER, COURT_LENGTH, COURT_WIDTH
from .sky_sphere import build_sky_sphere_world

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

    atlas_path = Path(__file__).resolve().parents[1] / "assets" / "fictional-sponsor-led-atlas-v1.png"
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
    sky_sphere = build_sky_sphere_world(scene, config)

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
        "sky_sphere": sky_sphere,
        "sky_sphere_active": False,
        "sky_sphere_strength_multiplier": 1.0,
        "config": config,
    }
