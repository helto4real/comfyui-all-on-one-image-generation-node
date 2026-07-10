"""FLUX.2 Klein 9B adapter scaffold."""

from __future__ import annotations

from typing import Any

try:
    from ..loaders import gguf_backend
    from ..services import inpaint as inpaint_service
    from ..services import pipeline
    from ..services import validation
    from ..services.registry import register_adapter
    from .base import BaseImageAdapter
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend
    from services import inpaint as inpaint_service
    from services import pipeline
    from services import validation
    from services.registry import register_adapter
    from adapters.base import BaseImageAdapter


def infer_edit_mode(reference_count: int, *, inpaint_enabled: bool = False) -> str:
    if inpaint_enabled and reference_count <= 0:
        return "inpaint"
    if inpaint_enabled and reference_count == 1:
        return "inpaint_single_reference"
    if inpaint_enabled:
        return "inpaint_multi_reference"
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
        if kwargs["steps"] <= 0:
            resolved["steps"] = 4 if variant == "distilled" else 50
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
        settings["edit_mode"] = infer_edit_mode(
            reference_count,
            inpaint_enabled=inpaint_config is not None,
        )
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
            warnings.append("mask is accepted for image 1 as a legacy no-op; use AIO Inpaint for inpaint.")
        duplicate_count = _duplicate_inpaint_reference_count(reference_inputs, inpaint_config)
        if duplicate_count > 0:
            warnings.append(
                "connected Flux reference image duplicates the AIO Inpaint source; "
                "using the cropped inpaint reference instead."
            )
        downscale_warning = inpaint_service.inpaint_full_frame_downscale_warning(
            inpaint_config,
            width=width,
            height=height,
        )
        if downscale_warning is not None:
            warnings.append(downscale_warning)
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
            batch_count=kwargs.get("batch_count", 1),
            steps=int(kwargs["settings"]["steps"]),
            cfg=float(kwargs["settings"]["cfg"]),
            sampler=kwargs["sampler"],
            scheduler=kwargs["scheduler"],
            settings=kwargs["settings"],
            lora_config=kwargs.get("lora_config"),
            loaded_model=kwargs.get("loaded_model"),
            loaded_clip=kwargs.get("loaded_clip"),
            reference_inputs=kwargs.get("reference_inputs"),
            inpaint_config=kwargs.get("inpaint_config"),
            inpaint_previews=kwargs.get("inpaint_previews"),
            decode_image=kwargs.get("decode_image", True),
            return_vae=kwargs.get("return_vae", False),
            second_pass_config=kwargs.get("second_pass_config"),
            second_pass_dimension_multiple=self.dimension_multiple,
            pid_capture_step=kwargs.get("pid_capture_step"),
            progress=progress,
        )


def _duplicate_inpaint_reference_count(reference_inputs: Any = None, inpaint_config: dict[str, Any] | None = None) -> int:
    if reference_inputs is None or inpaint_config is None:
        return 0
    inpaint_image = inpaint_config.get("image")
    if inpaint_image is None:
        return 0
    return sum(1 for image in (getattr(reference_inputs, "images", ()) or ()) if image is inpaint_image)
