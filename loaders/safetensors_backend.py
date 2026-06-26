"""Lazy path resolver for native ComfyUI safetensors model files."""

from __future__ import annotations

from pathlib import Path

try:
    from ..services.model_resolution import ResolvedModelPaths, strip_category_prefix
except ImportError:  # pragma: no cover - direct test imports
    from services.model_resolution import ResolvedModelPaths, strip_category_prefix


DIFFUSION_CATEGORIES = ("diffusion_models", "unet", "checkpoints")
TEXT_ENCODER_CATEGORIES = ("text_encoders", "clip")
VAE_CATEGORIES = ("vae",)

GGUF_DIFFUSION_CATEGORIES = (
    "unet_gguf",
    "model_gguf",
    "diffusion_models",
    "unet",
    "checkpoints",
)
GGUF_TEXT_ENCODER_CATEGORIES = ("clip_gguf", "text_encoders", "clip")
GGUF_VAE_CATEGORIES = ("vae_gguf", "vae")


def _folder_paths():
    import folder_paths  # type: ignore

    return folder_paths


def _resolve(filename: str, categories: tuple[str, ...]) -> Path:
    requested_category, requested_name = strip_category_prefix(filename)
    if requested_category in categories:
        search_paths = ((requested_category, requested_name),)
    elif requested_category:
        search_paths = tuple((category, filename) for category in categories)
    else:
        search_paths = tuple((category, requested_name) for category in categories)
    folder_paths = _folder_paths()
    for category, name in search_paths:
        try:
            path = folder_paths.get_full_path_or_raise(category, name)
        except Exception:
            continue
        if path:
            return Path(path)
    raise ValueError(
        f"Could not resolve '{filename}'. Check that it exists in: "
        f"{', '.join(categories)}."
    )


def diffusion_model_path(filename: str) -> Path:
    return _resolve(filename, DIFFUSION_CATEGORIES)


def text_encoder_path(filename: str) -> Path:
    return _resolve(filename, TEXT_ENCODER_CATEGORIES)


def vae_path(filename: str) -> Path:
    return _resolve(filename, VAE_CATEGORIES)


def resolve_paths(diffusion_model: str, text_encoder: str, vae: str) -> ResolvedModelPaths:
    return ResolvedModelPaths(
        diffusion_model=diffusion_model_path(diffusion_model),
        text_encoder=text_encoder_path(text_encoder),
        vae=vae_path(vae),
    )


def resolve_gguf_aware_paths(
    diffusion_model: str,
    text_encoder: str,
    vae: str,
) -> ResolvedModelPaths:
    return ResolvedModelPaths(
        diffusion_model=_resolve(diffusion_model, GGUF_DIFFUSION_CATEGORIES),
        text_encoder=_resolve(text_encoder, GGUF_TEXT_ENCODER_CATEGORIES),
        vae=_resolve(vae, GGUF_VAE_CATEGORIES),
    )
