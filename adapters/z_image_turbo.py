"""Z-Image Turbo adapter scaffold."""

from __future__ import annotations

from typing import Any

try:
    from ..loaders import gguf_backend
    from ..services import pipeline
    from ..services import validation
    from ..services.registry import register_adapter
    from .base import BaseImageAdapter
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend
    from services import pipeline
    from services import validation
    from services.registry import register_adapter
    from adapters.base import BaseImageAdapter


@register_adapter
class ZImageTurboAdapter(BaseImageAdapter):
    model_type = "z_image_turbo"
    profile_key = "z_image_turbo"
    version = "0.1.0"
    dimension_multiple = 16

    def resolve_settings(self, **kwargs) -> dict[str, Any]:
        resolved = super().resolve_settings(**kwargs)
        if resolved.get("force_steps"):
            resolved["steps"] = int(resolved["force_steps"])
        return resolved

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
        reference_inputs: Any = None,
        mask: Any = None,
    ) -> list[str]:
        profile = self.profile()
        validation.validate_required_model_names(
            profile.required_components, diffusion_model, text_encoder, vae
        )
        validation.validate_dimensions(
            width,
            height,
            validation.dimension_multiple_from_settings(settings, self.dimension_multiple),
        )
        validation.validate_reference_inputs(
            profile,
            reference_image=reference_image,
            reference_inputs=reference_inputs,
            mask=mask,
        )
        if not positive_prompt.strip():
            raise ValueError("positive_prompt is required.")
        warning = validation.validate_negative_prompt_policy(
            profile,
            negative_prompt,
            ignored_by_default=bool(settings.get("ignore_negative_prompt", True)),
        )
        validation.validate_gguf_available_for_models(
            gguf_backend.is_available(), diffusion_model, text_encoder, vae
        )
        return [warning] if warning else []

    def generate(self, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress.phase("resolving models")
        return pipeline.generate_z_image_turbo_t2i(
            diffusion_model=kwargs["diffusion_model"],
            text_encoder=kwargs["text_encoder"],
            vae=kwargs["vae"],
            positive_prompt=kwargs["positive_prompt"],
            width=kwargs["width"],
            height=kwargs["height"],
            seed=kwargs["seed"],
            steps=int(kwargs["settings"]["steps"]),
            cfg=float(kwargs["settings"]["cfg"]),
            sampler=kwargs["sampler"],
            scheduler=kwargs["scheduler"],
            settings=kwargs["settings"],
            lora_config=kwargs.get("lora_config"),
            loaded_model=kwargs.get("loaded_model"),
            loaded_clip=kwargs.get("loaded_clip"),
            decode_image=kwargs.get("decode_image", True),
            return_vae=kwargs.get("return_vae", False),
            pid_capture_step=kwargs.get("pid_capture_step"),
            progress=progress,
        )
