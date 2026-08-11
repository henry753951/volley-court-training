"""CLI and compatibility entry point for the modular Blender court generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from volleyball_synthetic import (
    DEFAULT_CONFIG_PATH,
    KEYPOINT_NAMES,
    KEYPOINT_WORLD,
    build_scene,
    configure_render,
    create_and_save_project,
    generate_dataset,
    load_randomization_config,
    randomize_scene,
)

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
