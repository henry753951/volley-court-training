"""Generate one dataset mixing balanced sky spheres with weighted virtual arenas."""

from __future__ import annotations

import argparse
import json
import math
import random
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
DEFAULT_OUTPUT = MODULE_DIRECTORY.parent / "datasets" / "court36-synthetic-combined-2000-v1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--sky-sphere-count", type=int, default=1000)
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


def _allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an exact total using largest remainders and stable tie-breaking."""

    if total < 0:
        raise ValueError("allocation total cannot be negative")
    active = {name: float(weight) for name, weight in weights.items() if float(weight) > 0.0}
    if total and not active:
        raise ValueError("at least one positive allocation weight is required")
    if not active:
        return {}
    weight_total = sum(active.values())
    raw = {name: total * weight / weight_total for name, weight in active.items()}
    allocated = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(allocated.values())
    remainder_order = sorted(active, key=lambda name: (-(raw[name] - allocated[name]), name))
    for offset in range(remaining):
        allocated[remainder_order[offset % len(remainder_order)]] += 1
    return dict(sorted(allocated.items()))


def _build_schedule(
    *,
    total_count: int,
    sky_sphere_count: int,
    asset_names: list[str],
    virtual_weights: dict[str, float],
    seed: int,
) -> tuple[list[tuple[str, str | None]], dict[str, int], dict[str, int]]:
    if total_count <= 0:
        raise ValueError("count must be greater than zero")
    if not 0 <= sky_sphere_count <= total_count:
        raise ValueError("sky-sphere-count must be between zero and count")
    if sky_sphere_count and sky_sphere_count < len(asset_names):
        raise ValueError("sky-sphere-count must render every configured sky sphere at least once")

    asset_counts = _allocate_counts(sky_sphere_count, {name: 1.0 for name in asset_names})
    virtual_counts = _allocate_counts(total_count - sky_sphere_count, virtual_weights)
    schedule: list[tuple[str, str | None]] = []
    for asset_name, count in asset_counts.items():
        schedule.extend(("sky_sphere", asset_name) for _ in range(count))
    for profile, count in virtual_counts.items():
        schedule.extend((profile, None) for _ in range(count))
    random.Random(seed ^ 0xA63E5F19).shuffle(schedule)
    return schedule, asset_counts, virtual_counts


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    asset_counts: dict[str, int],
    virtual_counts: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    samples = manifest["samples"]
    expected_profiles = {"sky_sphere": sum(asset_counts.values()), **virtual_counts}
    profile_counts = Counter(str(sample["arena_profile"]) for sample in samples)
    if dict(sorted(profile_counts.items())) != dict(sorted(expected_profiles.items())):
        raise RuntimeError(
            f"arena profile distribution differs from schedule: expected {expected_profiles}, got {dict(profile_counts)}"
        )

    rendered_assets = Counter(
        str(sample["sky_sphere_asset"])
        for sample in samples
        if sample["arena_profile"] == "sky_sphere"
    )
    if dict(sorted(rendered_assets.items())) != dict(sorted(asset_counts.items())):
        raise RuntimeError(
            f"sky sphere distribution differs from schedule: expected {asset_counts}, got {dict(rendered_assets)}"
        )

    invalid_sky = [
        sample["index"]
        for sample in samples
        if sample["arena_profile"] == "sky_sphere"
        and (
            int(sample["arena_background_count"]) != 0
            or int(sample["audience_count"]) != 0
            or int(sample["advertising_led_count"]) != 0
        )
    ]
    if invalid_sky:
        raise RuntimeError(f"generated 3D arena elements leaked into sky-sphere samples: {invalid_sky}")

    invalid_virtual = [
        sample["index"]
        for sample in samples
        if sample["arena_profile"] != "sky_sphere" and sample["sky_sphere_asset"] != "none"
    ]
    if invalid_virtual:
        raise RuntimeError(f"sky sphere leaked into virtual-arena samples: {invalid_virtual}")
    return dict(sorted(profile_counts.items())), dict(sorted(rendered_assets.items()))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)

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
    config["arena"]["forced_profile"] = ""
    config["arena"]["sky_sphere"]["assets"] = assets
    virtual_weights = {
        str(profile): float(settings["weight"])
        for profile, settings in config["arena"]["profiles"].items()
        if profile != "sky_sphere" and float(settings["weight"]) > 0.0
    }
    schedule, asset_counts, virtual_counts = _build_schedule(
        total_count=args.count,
        sky_sphere_count=args.sky_sphere_count,
        asset_names=asset_names,
        virtual_weights=virtual_weights,
        seed=args.seed,
    )

    state = build_scene(args.seed, config)
    state["arena_profile_sequence"] = [profile for profile, _ in schedule]
    state["sky_sphere_asset_by_sample_index"] = [asset for _, asset in schedule]
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
    profile_counts, rendered_asset_counts = _validate_manifest(
        manifest,
        asset_counts=asset_counts,
        virtual_counts=virtual_counts,
    )
    manifest["combined_arena_run"] = {
        "asset_directory": str(args.assets.resolve()),
        "schedule_seed": args.seed,
        "schedule_shuffled": True,
        "arena_profile_counts": profile_counts,
        "sky_sphere_asset_counts": rendered_asset_counts,
        "sky_sphere_calibration": calibration,
        "sky_sphere_generated_3d_arena_backgrounds": False,
        "sky_sphere_generated_3d_audience": False,
        "sky_sphere_generated_3d_advertising_led": False,
        "virtual_arena_profiles_use_generated_3d_elements": True,
        "all_other_randomization_from_yaml": True,
    }
    manifest_path = args.output.resolve() / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": args.count,
                "arena_profile_counts": profile_counts,
                "sky_sphere_asset_counts": rendered_asset_counts,
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
