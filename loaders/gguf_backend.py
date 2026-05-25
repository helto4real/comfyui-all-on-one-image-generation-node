"""Optional GGUF backend detection and path resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    from ..services.model_resolution import ResolvedModelPaths
    from . import safetensors_backend
except ImportError:  # pragma: no cover - direct test imports
    from services.model_resolution import ResolvedModelPaths
    from loaders import safetensors_backend


MISSING_GGUF_MESSAGE = (
    "GGUF support requires a compatible GGUF loader backend, such as ComfyUI-GGUF. "
    "Install it, restart ComfyUI, then retry with the .gguf file selected."
)


def _custom_node_dirs() -> list[Path]:
    try:
        import folder_paths  # type: ignore

        entries = folder_paths.folder_names_and_paths.get("custom_nodes", [])
        return [Path(item[0]) for item in entries if item]
    except Exception:
        return []


def is_available() -> bool:
    if importlib.util.find_spec("gguf") is not None:
        return True
    for custom_nodes_dir in _custom_node_dirs():
        for name in ("ComfyUI-GGUF", "comfyui-gguf"):
            if (custom_nodes_dir / name).exists():
                return True
    return False


def explain_missing() -> str:
    return MISSING_GGUF_MESSAGE


def resolve_paths(diffusion_model: str, text_encoder: str, vae: str) -> ResolvedModelPaths:
    return safetensors_backend.resolve_gguf_aware_paths(
        diffusion_model, text_encoder, vae
    )


def maybe_load_model(*args, **kwargs):
    if not is_available():
        raise ValueError(MISSING_GGUF_MESSAGE)
    raise NotImplementedError(
        "A compatible GGUF backend was detected, but automatic GGUF loading is not "
        "wired yet. Use the backend's dedicated loader nodes until this adapter is implemented."
    )
