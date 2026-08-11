"""Discovery and validation for calibrated arena panorama assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_yaml_mapping


DEFAULT_CALIBRATION_FILENAME = "arena-calibration.yaml"
DEFAULT_OVERRIDE_FILENAME = "arena-calibration-overrides.yaml"


def load_calibrated_sky_sphere_assets(
    asset_directory: Path | str,
    expected_count: int,
    calibration_filename: str = DEFAULT_CALIBRATION_FILENAME,
    override_filename: str | None = DEFAULT_OVERRIDE_FILENAME,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load all arena_*.png files and merge their reviewed projection calibration."""

    directory = Path(asset_directory).resolve()
    paths = sorted(directory.glob("arena_*.png"))
    if len(paths) != expected_count:
        raise ValueError(
            f"expected {expected_count} arena_*.png sky spheres in {directory}, found {len(paths)}"
        )
    calibration_path = directory / calibration_filename
    document, parser_name = load_yaml_mapping(calibration_path)
    calibrations = document.get("assets")
    if not isinstance(calibrations, dict):
        raise ValueError(f"{calibration_path} must contain an assets mapping")

    override_path = directory / override_filename if override_filename else None
    overrides: dict[str, Any] = {}
    override_parser: str | None = None
    if override_path is not None and override_path.exists():
        override_document, override_parser = load_yaml_mapping(override_path)
        raw_overrides = override_document.get("assets", {})
        if not isinstance(raw_overrides, dict):
            raise ValueError(f"{override_path} must contain an assets mapping")
        unknown_overrides = sorted(str(name) for name in raw_overrides if str(name) not in calibrations)
        if unknown_overrides:
            raise ValueError(f"unknown sky sphere calibration overrides: {unknown_overrides}")
        overrides = raw_overrides

    expected_names = {path.stem for path in paths}
    configured_names = {str(name) for name in calibrations}
    if configured_names != expected_names:
        missing = sorted(expected_names - configured_names)
        extra = sorted(configured_names - expected_names)
        raise ValueError(
            f"sky sphere calibration mismatch; missing={missing or 'none'}, extra={extra or 'none'}"
        )

    assets: dict[str, dict[str, Any]] = {}
    for path in paths:
        base_entry = calibrations[path.stem]
        if not isinstance(base_entry, dict):
            raise ValueError(f"calibration for {path.stem} must be a mapping")
        override_entry = overrides.get(path.stem, {})
        if not isinstance(override_entry, dict):
            raise ValueError(f"calibration override for {path.stem} must be a mapping")
        entry = {**base_entry, **override_entry}
        configured_image = str(entry.get("image", path.name))
        if configured_image != path.name:
            raise ValueError(
                f"calibration image for {path.stem} must be {path.name}, got {configured_image}"
            )
        mapping_scale = entry.get("mapping_scale", [1.0, 1.0, 1.0])
        mapping_location = entry.get("mapping_location", [0.0, 0.0, 0.0])
        if not isinstance(mapping_scale, (list, tuple)) or len(mapping_scale) != 3:
            raise ValueError(f"{path.stem}.mapping_scale must contain three values")
        if not isinstance(mapping_location, (list, tuple)) or len(mapping_location) != 3:
            raise ValueError(f"{path.stem}.mapping_location must contain three values")
        if any(float(value) <= 0.0 for value in mapping_scale):
            raise ValueError(f"{path.stem}.mapping_scale values must be positive")
        rotation_degrees = entry.get("rotation_degrees", [-180.0, 180.0])
        strength_multiplier = entry.get("strength_multiplier", [0.8, 1.0])
        for field_name, value in (
            ("rotation_degrees", rotation_degrees),
            ("strength_multiplier", strength_multiplier),
        ):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{path.stem}.{field_name} must contain two values")
            if float(value[0]) > float(value[1]):
                raise ValueError(f"{path.stem}.{field_name} minimum cannot exceed maximum")
        assets[path.stem] = {
            **entry,
            "stars": str(entry.get("stars", "★★★★★")),
            "weight": float(entry.get("weight", 1.0)),
            "path": str(path),
            "mapping_scale": [float(value) for value in mapping_scale],
            "mapping_location": [float(value) for value in mapping_location],
        }
    metadata = {
        "path": str(calibration_path),
        "version": int(document.get("version", 1)),
        "parser": parser_name,
        "method": str(document.get("method", "unspecified")),
        "overrides_path": str(override_path) if override_path is not None and override_path.exists() else None,
        "overrides_parser": override_parser,
    }
    return assets, metadata
