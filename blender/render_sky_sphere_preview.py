"""Render one forced sky-sphere frame and save an isolated experiment blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from generate_volleyball_dataset import (
    build_scene,
    configure_render,
    load_randomization_config,
    randomize_scene,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODULE_DIRECTORY / "randomization-config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--camera-mode", default="handheld_sideline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    config = load_randomization_config(args.config)
    config["arena"]["forced_profile"] = "sky_sphere"
    if args.camera_mode not in config["camera"]["modes"]:
        raise ValueError(f"unknown camera mode: {args.camera_mode}")
    for mode_name, mode_config in config["camera"]["modes"].items():
        mode_config["weight"] = 1 if mode_name == args.camera_mode else 0

    state = build_scene(args.seed, config)
    scene = state["scene"]
    configure_render(
        scene,
        config=config,
        width=args.width,
        height=args.height,
        samples=args.samples,
        engine="BLENDER_EEVEE",
    )
    preview = randomize_scene(state, args.seed)
    if preview["advertising_led_count"] != 0:
        raise RuntimeError("sky-sphere preview must not render generated advertising LED boards")

    output_path = args.output.resolve()
    blend_path = args.blend.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 94
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    active_image = state["sky_sphere"]["environment"].image
    if active_image is not None and not active_image.packed_file:
        active_image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    result: dict[str, object] = {
        "image": str(output_path),
        "blend": str(blend_path),
        "seed": args.seed,
        "preview": preview,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
