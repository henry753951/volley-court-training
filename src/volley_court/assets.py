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


DEFAULT_MODEL = ModelSpec(
    name="court-line-yolo26n-layout-v3",
    url=("https://assets.hsulab.net/models/volley-court-lines/v3/court-line-yolo26n-layout-v3.pt"),
    sha256="fb4abb0656d313fb5b6a3ec57d5b1531ac3b5052c14575dea10d4d06f46b71e4",
    filename="court-line-yolo26n-layout-v3.pt",
)

ROLLBACK_MODEL_V2 = ModelSpec(
    name="court-line-yolo26n-layout-v2",
    url="https://assets.hsulab.net/models/volley-court-lines/v2/court-line-yolo26n-layout-v2.pt",
    sha256="8fa56841200c5bc09635a2b26325a88e596a0f1198791ba5860af96ca41a0abd",
    filename="court-line-yolo26n-layout-v2.pt",
)

ROLLBACK_MODEL_V1 = ModelSpec(
    name="court-line-yolo26n-v3",
    url="https://assets.hsulab.net/models/volley-court-lines/v1/court-line-yolo26n-v3.pt",
    sha256="b0392c221978c87405f2646f41f14c1b66d4e7940d07c4a19c170b8321119e86",
    filename="court-line-yolo26n-v3.pt",
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
    if model is None or model == "v3" or model == DEFAULT_MODEL.name:
        return download_model(DEFAULT_MODEL)
    if model == "v2" or model == ROLLBACK_MODEL_V2.name:
        return download_model(ROLLBACK_MODEL_V2)
    if model == "v1" or model == ROLLBACK_MODEL_V1.name:
        return download_model(ROLLBACK_MODEL_V1)
    if isinstance(model, ModelSpec):
        return download_model(model)
    path = Path(model).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    return path
