from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    filename: str


LEGACY_MODEL = ModelSpec(
    name="court-line-yolo26n-v3",
    url=("https://assets.hsulab.net/models/volley-court-lines/v1/court-line-yolo26n-v3.pt"),
    sha256="b0392c221978c87405f2646f41f14c1b66d4e7940d07c4a19c170b8321119e86",
    filename="court-line-yolo26n-v3.pt",
)

DEFAULT_MODEL = ModelSpec(
    name="court-line-yolo26n-semantic-v4",
    url=(
        "https://assets.hsulab.net/models/volley-court-lines/v2/court-line-yolo26n-semantic-v4.pt"
    ),
    sha256="b4aed936446e262518c927bf17dcf04877c7ba0f96e0db7687e84c3b2ea12b2b",
    filename="court-line-yolo26n-semantic-v4.pt",
)


class ModelIntegrityError(RuntimeError):
    pass


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_dir() -> Path:
    configured = os.environ.get("VOLLEY_COURT_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and (local_app_data := os.environ.get("LOCALAPPDATA")):
        return Path(local_app_data) / "HSULab" / "volley-court-lines" / "Cache"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    return (
        Path(xdg_cache).expanduser() / "volley-court-lines"
        if xdg_cache
        else Path.home() / ".cache" / "volley-court-lines"
    )


def download_model(
    spec: ModelSpec = DEFAULT_MODEL,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / spec.filename
    if destination.is_file() and not force:
        if sha256sum(destination) == spec.sha256:
            return destination
        raise ModelIntegrityError(
            f"cached model checksum mismatch: {destination}; pass force=True to replace it"
        )
    request = Request(spec.url, headers={"User-Agent": "volley-court-lines/0.1"})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{spec.filename}.",
            suffix=".download",
            dir=root,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256sum(temporary)
        if actual != spec.sha256:
            raise ModelIntegrityError(
                f"downloaded model checksum mismatch: expected {spec.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def resolve_model_path(model: str | Path | ModelSpec | None = None) -> Path:
    configured = os.environ.get("VOLLEY_COURT_MODEL")
    if model is None and configured:
        model = configured
    if model is None or model == "v2" or model == DEFAULT_MODEL.name:
        return download_model(DEFAULT_MODEL)
    if model == "v1" or model == LEGACY_MODEL.name:
        return download_model(LEGACY_MODEL)
    if isinstance(model, ModelSpec):
        return download_model(model)
    path = Path(model).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    return path
