"""Render configuration and broadcast/camera-quality post-processing."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from .config import _randint, _uniform, _weighted_entry

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
