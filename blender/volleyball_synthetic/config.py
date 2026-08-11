"""YAML loading and weighted randomization primitives."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "randomization-config.yaml"

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

def load_yaml_mapping(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load a YAML mapping with PyYAML or the bundled mapping-only parser."""

    yaml_path = Path(path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _load_yaml_without_dependency(yaml_path), "builtin-yaml-subset"
    loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return loaded, "PyYAML"

def load_randomization_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    config, parser_name = load_yaml_mapping(config_path)
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
