"""Krea 2 adapter."""

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
class Krea2Adapter(BaseImageAdapter):
    model_type = "krea2"
    profile_key = "krea2"
    version = "0.1.0"
    dimension_multiple = 16

    def resolve_settings(self, **kwargs) -> dict[str, Any]:
        resolved = super().resolve_settings(**kwargs)
        resolved["max_length"] = pipeline.normalize_krea2_max_length(resolved.get("max_length"))
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
        inpaint_config: dict[str, Any] | None = None,
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
        pipeline.normalize_krea2_max_length(settings.get("max_length"))
        if not positive_prompt.strip():
            raise ValueError("positive_prompt is required.")
        validation.validate_reference_inputs(
            profile,
            reference_image=reference_image,
            reference_inputs=reference_inputs,
            mask=mask,
        )
        validation.validate_inpaint_config(profile, inpaint_config)
        use_zero_negative_conditioning = bool(settings.get("use_zero_negative_conditioning", True))
        warning = (
            validation.validate_negative_prompt_policy(
                profile,
                negative_prompt,
                ignored_by_default=True,
            )
            if use_zero_negative_conditioning
            else None
        )
        validation.validate_gguf_available_for_models(
            gguf_backend.is_available(), diffusion_model, text_encoder, vae
        )
        return [warning] if warning else []

    def generate(self, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress.phase("resolving models")
        return pipeline.generate_krea2_t2i(
            diffusion_model=kwargs["diffusion_model"],
            text_encoder=kwargs["text_encoder"],
            vae=kwargs["vae"],
            positive_prompt=kwargs["positive_prompt"],
            negative_prompt=kwargs["negative_prompt"],
            width=kwargs["width"],
            height=kwargs["height"],
            seed=kwargs["seed"],
            batch_count=kwargs.get("batch_count", 1),
            steps=int(kwargs["settings"]["steps"]),
            cfg=float(kwargs["settings"]["cfg"]),
            sampler=kwargs["sampler"],
            scheduler=kwargs["scheduler"],
            settings=kwargs["settings"],
            lora_config=kwargs.get("lora_config"),
            loaded_model=kwargs.get("loaded_model"),
            loaded_clip=kwargs.get("loaded_clip"),
            inpaint_config=kwargs.get("inpaint_config"),
            inpaint_previews=kwargs.get("inpaint_previews"),
            decode_image=kwargs.get("decode_image", True),
            return_vae=kwargs.get("return_vae", False),
            second_pass_config=kwargs.get("second_pass_config"),
            second_pass_dimension_multiple=self.dimension_multiple,
            pid_capture_step=kwargs.get("pid_capture_step"),
            progress=progress,
        )
