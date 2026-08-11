"""Launch an in-Blender live calibration panel for arena sky-sphere panoramas."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import bpy

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from volleyball_synthetic import (
    DEFAULT_CONFIG_PATH,
    build_scene,
    configure_render,
    load_calibrated_sky_sphere_assets,
    load_randomization_config,
    randomize_scene,
)


DEFAULT_ASSET_DIRECTORY = MODULE_DIRECTORY / "assets" / "sky_spheres" / "blackfloor-v1"
OVERRIDE_FILENAME = "arena-calibration-overrides.yaml"

_STATE: dict[str, Any] | None = None
_ASSETS: dict[str, dict[str, Any]] = {}
_ASSET_DIRECTORY = DEFAULT_ASSET_DIRECTORY
_SUPPRESS_UPDATES = False


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_DIRECTORY)
    parser.add_argument("--expected-assets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args(argv)


def _asset_items(_self: Any, _context: bpy.types.Context) -> list[tuple[str, str, str]]:
    return [(name, name, str(_ASSETS[name].get("arena_type", "unknown"))) for name in _ASSETS]


def _float_triplet(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return fallback
    return tuple(float(component) for component in value)


def _float_pair(value: Any, fallback: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return fallback
    return float(value[0]), float(value[1])


def _store_properties(scene: bpy.types.Scene) -> None:
    props = scene.court_sky_calibration
    asset_name = str(props.active_asset)
    if not asset_name or asset_name not in _ASSETS:
        return
    entry = _ASSETS[asset_name]
    entry["yaw_offset_degrees"] = float(props.yaw_offset_degrees)
    entry["rotation_degrees"] = [float(props.rotation_min_degrees), float(props.rotation_max_degrees)]
    entry["mapping_location"] = [
        float(props.location_x),
        float(props.location_y),
        float(props.horizon_offset_z),
    ]
    entry["mapping_scale"] = [
        float(props.scale_x),
        float(props.scale_y),
        float(props.apparent_distance_scale),
    ]
    entry["strength_multiplier"] = [float(props.strength_min), float(props.strength_max)]


def _apply_live_preview(scene: bpy.types.Scene) -> None:
    if _STATE is None:
        return
    props = scene.court_sky_calibration
    asset_name = str(props.active_asset)
    if asset_name not in _ASSETS:
        return
    _store_properties(scene)
    sky_state = _STATE["sky_sphere"]
    entry = _ASSETS[asset_name]
    mapping = sky_state["mapping"]
    orientation = _STATE["config"]["arena"]["sky_sphere"].get(
        "orientation_degrees", [180.0, 0.0, 0.0]
    )
    mapping.inputs["Location"].default_value = tuple(float(v) for v in entry["mapping_location"])
    mapping.inputs["Scale"].default_value = tuple(float(v) for v in entry["mapping_scale"])
    mapping.inputs["Rotation"].default_value = (
        math.radians(float(orientation[0])),
        math.radians(float(orientation[1])),
        math.radians(
            float(orientation[2])
            + float(entry["yaw_offset_degrees"])
            + float(props.preview_rotation_degrees)
        ),
    )
    sky_state["environment"].image = sky_state["images"][asset_name]
    background = sky_state["background"]
    node_tree = scene.world.node_tree
    for link in list(background.inputs["Color"].links):
        node_tree.links.remove(link)
    node_tree.links.new(sky_state["environment"].outputs["Color"], background.inputs["Color"])
    base_world_strength = sum(_STATE["config"]["lighting"]["world_strength"]) / 2.0
    preview_multiplier = (float(props.strength_min) + float(props.strength_max)) / 2.0
    background.inputs["Strength"].default_value = base_world_strength * preview_multiplier
    scene["sky_calibration_asset"] = asset_name
    scene["sky_calibration_status"] = "Live preview updated; press Save Overrides to persist."
    scene.world.update_tag()


def _load_asset_properties(scene: bpy.types.Scene, asset_name: str) -> None:
    global _SUPPRESS_UPDATES
    entry = _ASSETS[asset_name]
    location = _float_triplet(entry.get("mapping_location"), (0.0, 0.0, 0.0))
    scale = _float_triplet(entry.get("mapping_scale"), (1.0, 1.0, 1.0))
    rotation = _float_pair(entry.get("rotation_degrees"), (-180.0, 180.0))
    strength = _float_pair(entry.get("strength_multiplier"), (0.8, 1.0))
    props = scene.court_sky_calibration
    _SUPPRESS_UPDATES = True
    try:
        props.active_asset = asset_name
        props.yaw_offset_degrees = float(entry.get("yaw_offset_degrees", 0.0))
        props.preview_rotation_degrees = 0.0
        props.rotation_min_degrees, props.rotation_max_degrees = rotation
        props.location_x, props.location_y, props.horizon_offset_z = location
        props.scale_x, props.scale_y, props.apparent_distance_scale = scale
        props.strength_min, props.strength_max = strength
    finally:
        _SUPPRESS_UPDATES = False
    _apply_live_preview(scene)


def _property_updated(_self: Any, context: bpy.types.Context) -> None:
    if not _SUPPRESS_UPDATES and context.scene is not None:
        _apply_live_preview(context.scene)


def _asset_updated(self: Any, context: bpy.types.Context) -> None:
    if not _SUPPRESS_UPDATES and context.scene is not None and self.active_asset in _ASSETS:
        _load_asset_properties(context.scene, str(self.active_asset))


class CourtSkyCalibrationProperties(bpy.types.PropertyGroup):
    active_asset: bpy.props.EnumProperty(name="Arena", items=_asset_items, update=_asset_updated)
    yaw_offset_degrees: bpy.props.FloatProperty(
        name="Yaw correction",
        description="Persistent left/right panorama alignment",
        min=-180.0,
        max=180.0,
        update=_property_updated,
    )
    preview_rotation_degrees: bpy.props.FloatProperty(
        name="Preview rotation",
        description="Temporary rotation for checking other directions; not saved",
        min=-180.0,
        max=180.0,
        update=_property_updated,
    )
    rotation_min_degrees: bpy.props.FloatProperty(name="Dataset yaw min", min=-180.0, max=180.0, update=_property_updated)
    rotation_max_degrees: bpy.props.FloatProperty(name="Dataset yaw max", min=-180.0, max=180.0, update=_property_updated)
    location_x: bpy.props.FloatProperty(name="Mapping offset X", min=-1.0, max=1.0, precision=3, update=_property_updated)
    location_y: bpy.props.FloatProperty(name="Mapping offset Y", min=-1.0, max=1.0, precision=3, update=_property_updated)
    horizon_offset_z: bpy.props.FloatProperty(
        name="Horizon offset Z",
        description="Positive values move the panorama floor boundary downward",
        min=-0.25,
        max=0.25,
        precision=3,
        update=_property_updated,
    )
    scale_x: bpy.props.FloatProperty(name="Projection scale X", min=0.5, max=2.0, precision=3, update=_property_updated)
    scale_y: bpy.props.FloatProperty(name="Projection scale Y", min=0.5, max=2.0, precision=3, update=_property_updated)
    apparent_distance_scale: bpy.props.FloatProperty(
        name="Apparent distance",
        description="Higher values compress vertical venue scale and make it appear farther away",
        min=0.5,
        max=2.0,
        precision=3,
        update=_property_updated,
    )
    strength_min: bpy.props.FloatProperty(name="Brightness min", min=0.1, max=3.0, precision=2, update=_property_updated)
    strength_max: bpy.props.FloatProperty(name="Brightness max", min=0.1, max=3.0, precision=2, update=_property_updated)


class COURTSKY_OT_previous_asset(bpy.types.Operator):
    bl_idname = "court_sky.previous_asset"
    bl_label = "Previous"

    def execute(self, context: bpy.types.Context) -> set[str]:
        names = list(_ASSETS)
        current = names.index(context.scene.court_sky_calibration.active_asset)
        _load_asset_properties(context.scene, names[(current - 1) % len(names)])
        return {"FINISHED"}


class COURTSKY_OT_next_asset(bpy.types.Operator):
    bl_idname = "court_sky.next_asset"
    bl_label = "Next"

    def execute(self, context: bpy.types.Context) -> set[str]:
        names = list(_ASSETS)
        current = names.index(context.scene.court_sky_calibration.active_asset)
        _load_asset_properties(context.scene, names[(current + 1) % len(names)])
        return {"FINISHED"}


class COURTSKY_OT_camera_view(bpy.types.Operator):
    bl_idname = "court_sky.camera_view"
    bl_label = "Camera View + Lock"

    def execute(self, context: bpy.types.Context) -> set[str]:
        if context.area is not None and context.area.type == "VIEW_3D":
            context.space_data.region_3d.view_perspective = "CAMERA"
            context.space_data.lock_camera = True
            context.space_data.shading.type = "RENDERED"
        return {"FINISHED"}


def _write_overrides(path: Path) -> None:
    lines = [
        "version: 1",
        'notes: "Generated by blender/sky_sphere_calibrator.py; values override arena-calibration.yaml"',
        "assets:",
    ]
    for asset_name, entry in _ASSETS.items():
        location = [float(value) for value in entry["mapping_location"]]
        scale = [float(value) for value in entry["mapping_scale"]]
        rotation = [float(value) for value in entry["rotation_degrees"]]
        strength = [float(value) for value in entry["strength_multiplier"]]
        lines.extend(
            (
                f"  {asset_name}:",
                f"    yaw_offset_degrees: {float(entry['yaw_offset_degrees']):.4f}",
                f"    rotation_degrees: [{rotation[0]:.4f}, {rotation[1]:.4f}]",
                f"    mapping_location: [{location[0]:.4f}, {location[1]:.4f}, {location[2]:.4f}]",
                f"    mapping_scale: [{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}]",
                f"    strength_multiplier: [{strength[0]:.4f}, {strength[1]:.4f}]",
            )
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(path)


class COURTSKY_OT_save_overrides(bpy.types.Operator):
    bl_idname = "court_sky.save_overrides"
    bl_label = "Save Overrides YAML"

    def execute(self, context: bpy.types.Context) -> set[str]:
        _store_properties(context.scene)
        invalid = [
            asset_name
            for asset_name, entry in _ASSETS.items()
            if float(entry["rotation_degrees"][0]) > float(entry["rotation_degrees"][1])
            or float(entry["strength_multiplier"][0]) > float(entry["strength_multiplier"][1])
        ]
        if invalid:
            self.report({"ERROR"}, f"Minimum exceeds maximum for: {', '.join(invalid)}")
            return {"CANCELLED"}
        output_path = _ASSET_DIRECTORY / OVERRIDE_FILENAME
        _write_overrides(output_path)
        context.scene["sky_calibration_status"] = f"Saved: {output_path}"
        self.report({"INFO"}, f"Saved sky-sphere overrides to {output_path}")
        return {"FINISHED"}


class COURTSKY_PT_calibration(bpy.types.Panel):
    bl_label = "Court Sky Calibration"
    bl_idname = "COURTSKY_PT_calibration"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Court Calib"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        props = context.scene.court_sky_calibration
        layout.prop(props, "active_asset")
        row = layout.row(align=True)
        row.operator("court_sky.previous_asset")
        row.operator("court_sky.next_asset")
        layout.operator("court_sky.camera_view", icon="CAMERA_DATA")
        layout.separator()
        layout.label(text="Rotation")
        layout.prop(props, "yaw_offset_degrees")
        layout.prop(props, "preview_rotation_degrees")
        row = layout.row(align=True)
        row.prop(props, "rotation_min_degrees")
        row.prop(props, "rotation_max_degrees")
        layout.separator()
        layout.label(text="Horizon / apparent distance")
        layout.prop(props, "horizon_offset_z")
        layout.prop(props, "apparent_distance_scale")
        row = layout.row(align=True)
        row.prop(props, "scale_x")
        row.prop(props, "scale_y")
        layout.separator()
        layout.label(text="Advanced mapping offset")
        row = layout.row(align=True)
        row.prop(props, "location_x")
        row.prop(props, "location_y")
        layout.separator()
        layout.label(text="Dataset brightness multiplier")
        row = layout.row(align=True)
        row.prop(props, "strength_min")
        row.prop(props, "strength_max")
        layout.separator()
        layout.operator("court_sky.save_overrides", icon="FILE_TICK")
        layout.label(text=str(context.scene.get("sky_calibration_status", "")), icon="INFO")


CLASSES = (
    CourtSkyCalibrationProperties,
    COURTSKY_OT_previous_asset,
    COURTSKY_OT_next_asset,
    COURTSKY_OT_camera_view,
    COURTSKY_OT_save_overrides,
    COURTSKY_PT_calibration,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.court_sky_calibration = bpy.props.PointerProperty(type=CourtSkyCalibrationProperties)


def _activate_camera_view() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.show_region_ui = True
            space.region_3d.view_perspective = "CAMERA"
            space.lock_camera = True
            space.shading.type = "RENDERED"
    return None


def main(argv: list[str] | None = None) -> int:
    global _STATE, _ASSETS, _ASSET_DIRECTORY
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)
    _ASSET_DIRECTORY = args.assets.resolve()
    config = load_randomization_config(args.config)
    assets, _metadata = load_calibrated_sky_sphere_assets(_ASSET_DIRECTORY, args.expected_assets)
    _ASSETS = assets
    config["arena"]["forced_profile"] = "sky_sphere"
    config["arena"]["sky_sphere"]["assets"] = assets
    config["camera"]["requested_partial_probability"] = 0.0
    for mode_name, mode in config["camera"]["modes"].items():
        mode["weight"] = 1 if mode_name == "championship_main" else 0
    _STATE = build_scene(args.seed, config)
    configure_render(_STATE["scene"], config=config, width=1280, height=720, samples=8, engine="BLENDER_EEVEE_NEXT")
    randomize_scene(_STATE, args.seed)
    register()
    first_asset = next(iter(_ASSETS))
    _load_asset_properties(_STATE["scene"], first_asset)
    _STATE["scene"]["sky_calibration_status"] = "Adjust sliders, then Save Overrides YAML."
    if not bpy.app.background:
        bpy.app.timers.register(_activate_camera_view, first_interval=0.5)
    return 0


if __name__ == "__main__":
    main()
