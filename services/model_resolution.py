"""Model filename/path resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedModelPaths:
    diffusion_model: Path
    text_encoder: Path
    vae: Path


def strip_category_prefix(value: str) -> tuple[str | None, str]:
    if "/" not in value:
        return None, value
    category, name = value.split("/", 1)
    return category, name


def infer_model_format(value: str) -> str:
    _, name = strip_category_prefix(value)
    if Path(name).suffix.lower() == ".gguf":
        return "gguf"
    return "safetensors"
