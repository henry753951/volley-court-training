from __future__ import annotations

from pathlib import Path

from volley_court import assets


def test_model_aliases_keep_v1_as_rollback(monkeypatch) -> None:
    monkeypatch.setattr(
        assets,
        "download_model",
        lambda spec=assets.DEFAULT_MODEL: Path(spec.filename),
    )

    assert assets.resolve_model_path().name == "court-line-yolo26n-semantic-v4.pt"
    assert assets.resolve_model_path("v2").name == "court-line-yolo26n-semantic-v4.pt"
    assert assets.resolve_model_path("v1").name == "court-line-yolo26n-v3.pt"
