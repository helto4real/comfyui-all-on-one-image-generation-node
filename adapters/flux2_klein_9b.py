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


def infer_edit_mode(reference_count: int) -> str:
    if reference_count <= 0:
        return "text_to_image"
    if reference_count == 1:
        return "single_reference"
    return "multi_reference"


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
        if not positive_prompt.strip():
            raise ValueError("positive_prompt is required.")
        reference_count = int(getattr(reference_inputs, "count", 0))
        if reference_image is not None:
            reference_count += 1
        if reference_count > 4:
            raise ValueError("FLUX.2 Klein supports at most four connected reference images.")
        settings["edit_mode"] = infer_edit_mode(reference_count)
        validation.validate_reference_inputs(
            profile,
            reference_image=reference_image,
            reference_inputs=reference_inputs,
            mask=mask,
        )
        validation.validate_inpaint_config(profile, inpaint_config)
        validation.validate_gguf_available_for_models(
            gguf_backend.is_available(), diffusion_model, text_encoder, vae
        )
        warnings: list[str] = []
        active_mask = mask if mask is not None else getattr(reference_inputs, "mask", None)
        if active_mask is not None:
            warnings.append("mask is accepted for image 1, but inpaint behavior is not implemented yet.")
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
            loaded_model=kwargs.get("loaded_model"),
            loaded_clip=kwargs.get("loaded_clip"),
            reference_inputs=kwargs.get("reference_inputs"),
            decode_image=kwargs.get("decode_image", True),
            return_vae=kwargs.get("return_vae", False),
            pid_capture_step=kwargs.get("pid_capture_step"),
            progress=progress,
        )
