from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from volley_court.model import load_court_line_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate two compatible court checkpoints for a validation-gated model soup"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Candidate weight in [0, 1]")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between zero and one")
    base_model, _ = load_court_line_checkpoint(args.base, device=torch.device("cpu"))
    candidate_model, _ = load_court_line_checkpoint(args.candidate, device=torch.device("cpu"))
    base_state = base_model.state_dict()
    candidate_state = candidate_model.state_dict()
    if base_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint model structures are incompatible")
    mixed: dict[str, torch.Tensor] = {}
    for key, candidate_value in candidate_state.items():
        base_value = base_state[key]
        mixed[key] = (
            torch.lerp(base_value, candidate_value, args.alpha)
            if torch.is_floating_point(candidate_value)
            else candidate_value.clone()
        )
    payload: dict[str, Any] = torch.load(
        args.candidate,
        map_location="cpu",
        weights_only=False,
    )
    payload["model"] = mixed
    payload["optimizer"] = {}
    payload["scheduler"] = {}
    payload["scaler"] = {}
    payload["interpolation"] = {
        "base": str(args.base.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_alpha": args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
