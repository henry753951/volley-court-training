"""Generate a balanced dataset from a directory of arena sky-sphere panoramas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from volleyball_synthetic import (
    DEFAULT_CONFIG_PATH,
    build_scene,
    configure_render,
    generate_dataset,
    load_calibrated_sky_sphere_assets,
    load_randomization_config,
)


DEFAULT_ASSET_DIRECTORY = MODULE_DIRECTORY / "assets" / "sky_spheres" / "blackfloor-v1"
DEFAULT_OUTPUT = MODULE_DIRECTORY.parent / "datasets" / "court36-synthetic-skysphere-blackfloor-50-v1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--expected-assets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--resolution-x", type=int)
    parser.add_argument("--resolution-y", type=int)
    parser.add_argument("--minimum-resolution-y", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"),
    )
    parser.add_argument("--fixed-capture-quality", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_sky_sphere_manifest(
    manifest: dict[str, Any],
    asset_names: list[str],
) -> dict[str, int]:
    samples = manifest["samples"]
    invalid = [
        sample["index"]
        for sample in samples
        if sample["arena_profile"] != "sky_sphere"
        or int(sample["arena_background_count"]) != 0
        or int(sample["audience_count"]) != 0
        or int(sample["advertising_led_count"]) != 0
    ]
    if invalid:
        raise RuntimeError(f"sky-sphere-only constraints failed for sample indices: {invalid}")
    counts = Counter(str(sample["sky_sphere_asset"]) for sample in samples)
    if set(counts) != set(asset_names):
        raise RuntimeError(f"not every configured sky sphere was rendered: {dict(counts)}")
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError(f"sky sphere distribution is not balanced: {dict(counts)}")
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)
    if args.count < args.expected_assets:
        raise ValueError("count must be at least expected-assets so every arena is rendered")

    config = load_randomization_config(args.config)
    generation = config["generation"]
    resolution_x = int(generation["resolution_x"] if args.resolution_x is None else args.resolution_x)
    resolution_y = int(generation["resolution_y"] if args.resolution_y is None else args.resolution_y)
    minimum_resolution_y = int(
        generation["minimum_resolution_y"]
        if args.minimum_resolution_y is None
        else args.minimum_resolution_y
    )
    samples = int(generation["samples"] if args.samples is None else args.samples)
    engine = str(generation["engine"] if args.engine is None else args.engine)
    capture_randomization = bool(generation["capture_randomization"]) and not args.fixed_capture_quality

    assets, calibration = load_calibrated_sky_sphere_assets(args.assets, args.expected_assets)
    asset_names = list(assets)
    config["arena"]["forced_profile"] = "sky_sphere"
    config["arena"]["sky_sphere"]["assets"] = assets

    state = build_scene(args.seed, config)
    state["sky_sphere_asset_sequence"] = asset_names
    configure_render(
        state["scene"],
        config=config,
        width=resolution_x,
        height=resolution_y,
        samples=samples,
        engine=engine,
    )
    manifest = generate_dataset(
        state,
        args.output,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
        minimum_resolution_y=minimum_resolution_y,
        randomize_capture_quality=capture_randomization,
    )
    asset_counts = _validate_sky_sphere_manifest(manifest, asset_names)
    manifest["sky_sphere_run"] = {
        "asset_directory": str(args.assets.resolve()),
        "asset_order": asset_names,
        "asset_counts": asset_counts,
        "calibration": calibration,
        "generated_3d_arena_backgrounds": False,
        "generated_3d_audience": False,
        "generated_3d_advertising_led": False,
        "all_other_randomization_from_yaml": True,
    }
    manifest_path = args.output.resolve() / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": args.count,
                "asset_counts": asset_counts,
                "resolution": manifest["resolution"],
                "capture_randomization": capture_randomization,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
