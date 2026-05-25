"""FLUX.2 Klein 9B adapter scaffold."""

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
class Flux2Klein9BAdapter(BaseImageAdapter):
    model_type = "flux2_klein_9b"
    profile_key = "flux2_klein_9b"
    version = "0.1.0"
    dimension_multiple = 16

    def resolve_settings(self, **kwargs) -> dict[str, Any]:
        resolved = super().resolve_settings(**kwargs)
        variant = resolved.get("variant", "distilled")
        if kwargs["steps"] <= 0 and variant == "distilled":
            resolved["steps"] = 4
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
        validation.validate_dimensions(width, height, self.dimension_multiple)
        if not positive_prompt.strip():
            raise ValueError("positive_prompt is required.")
        edit_mode = settings.get("edit_mode", "text_to_image")
        reference_count = int(getattr(reference_inputs, "count", 0))
        if reference_image is not None:
            reference_count += 1
        if edit_mode == "single_reference" and reference_count != 1:
            raise ValueError(
                "single_reference mode requires exactly one connected reference image."
            )
        if edit_mode == "multi_reference" and not 1 <= reference_count <= 4:
            raise ValueError(
                "multi_reference mode requires between one and four connected "
                "reference images."
            )
        validation.validate_reference_inputs(
            profile,
            reference_image=reference_image,
            reference_inputs=reference_inputs,
            mask=mask,
        )
        validation.validate_gguf_available_for_models(
            gguf_backend.is_available(), diffusion_model, text_encoder, vae
        )
        warnings: list[str] = []
        active_mask = mask if mask is not None else getattr(reference_inputs, "mask", None)
        if active_mask is not None:
            warnings.append("mask is accepted for image 1, but inpaint behavior is not implemented yet.")
        if edit_mode == "text_to_image" and reference_count:
            warnings.append(
                "reference images were connected with edit_mode='text_to_image'; "
                "using the reference conditioning path."
            )
        return warnings

    def generate(self, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress.phase("resolving models")
        return pipeline.generate_flux2_klein_t2i(
            diffusion_model=kwargs["diffusion_model"],
            text_encoder=kwargs["text_encoder"],
            vae=kwargs["vae"],
            positive_prompt=kwargs["positive_prompt"],
            negative_prompt=kwargs["negative_prompt"],
            width=kwargs["width"],
            height=kwargs["height"],
            seed=kwargs["seed"],
            steps=int(kwargs["settings"]["steps"]),
            cfg=float(kwargs["settings"]["cfg"]),
            sampler=kwargs["sampler"],
            scheduler=kwargs["scheduler"],
            settings=kwargs["settings"],
            lora_config=kwargs.get("lora_config"),
            reference_inputs=kwargs.get("reference_inputs"),
            progress=progress,
        )
