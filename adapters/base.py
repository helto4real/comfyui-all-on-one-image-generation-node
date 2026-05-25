"""Adapter contract for model-family generation implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    from ..services.profiles import ModelProfile
    from ..services.registry import get_profile
except ImportError:  # pragma: no cover - direct test imports
    from services.profiles import ModelProfile
    from services.registry import get_profile


class BaseImageAdapter(ABC):
    model_type: str = ""
    profile_key: str = ""
    version: str = "0.1.0"
    dimension_multiple: int = 8

    def profile(self) -> ModelProfile:
        return get_profile(self.model_type)

    def resolve_settings(
        self,
        *,
        model_settings: dict[str, Any] | None,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
    ) -> dict[str, Any]:
        profile = self.profile()
        resolved = dict(model_settings or {})
        resolved.setdefault("family", self.model_type)
        resolved["width"] = width or profile.default_width
        resolved["height"] = height or profile.default_height
        resolved["steps"] = steps if steps > 0 else profile.default_steps
        resolved["cfg"] = cfg if cfg > 0.0 else profile.default_cfg
        resolved["sampler"] = sampler if sampler != "auto" else profile.default_sampler
        resolved["scheduler"] = scheduler if scheduler != "auto" else profile.default_scheduler
        return resolved

    @abstractmethod
    def validate_inputs(
        self,
        *,
        diffusion_model: str,
        text_encoder: str,
        vae: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        settings: dict[str, Any],
        reference_image: Any = None,
        mask: Any = None,
    ) -> list[str]:
        """Validate adapter-specific inputs and return non-fatal warnings."""

    @abstractmethod
    def generate(
        self,
        *,
        diffusion_model: str,
        text_encoder: str,
        vae: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        settings: dict[str, Any],
        sampler: str,
        scheduler: str,
        reference_image: Any = None,
        mask: Any = None,
        lora_config: dict[str, Any] | None = None,
        progress: Any = None,
    ):
        """Generate and return (image, latent)."""
