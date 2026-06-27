"""Ideogram 4 adapter."""

from __future__ import annotations

from typing import Any

try:
    from ..loaders import gguf_backend
    from ..services import pipeline
    from ..services import validation
    from ..services.model_resolution import infer_model_format
    from ..services.registry import register_adapter
    from .base import BaseImageAdapter
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend
    from services import pipeline
    from services import validation
    from services.model_resolution import infer_model_format
    from services.registry import register_adapter
    from adapters.base import BaseImageAdapter


@register_adapter
class Ideogram4Adapter(BaseImageAdapter):
    model_type = "ideogram4"
    profile_key = "ideogram4"
    version = "0.1.0"
    dimension_multiple = 16

    def resolve_settings(self, **kwargs) -> dict[str, Any]:
        model_settings = dict(kwargs.get("model_settings") or {})
        resolved = super().resolve_settings(**kwargs)
        if kwargs["steps"] <= 0:
            resolved["steps"] = int(resolved.get("preset_steps", resolved["steps"]))
        resolved["cfg"] = float(resolved.get("dual_cfg", resolved["cfg"]))
        if kwargs["scheduler"] == "auto":
            resolved["scheduler"] = str(model_settings.get("scheduler", resolved.get("scheduler", "ideogram4")))
        if kwargs["sampler"] == "auto":
            resolved["sampler"] = "euler"
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
        if width < 256 or height < 256 or width > 2048 or height > 2048:
            raise ValueError("Ideogram 4 width and height must be between 256 and 2048.")
        aspect = max(width / height, height / width)
        if aspect > 6.0:
            raise ValueError("Ideogram 4 aspect ratio must not exceed 6:1.")
        run_unconditional_model = bool(settings.get("run_unconditional_model", True))
        unconditional_model = str(settings.get("unconditional_model", "")).strip()
        if run_unconditional_model and not unconditional_model:
            raise ValueError("unconditional_model is required for Ideogram 4.")
        validation.validate_reference_inputs(
            profile,
            reference_image=reference_image,
            reference_inputs=reference_inputs,
            mask=mask,
        )
        validation.validate_inpaint_config(profile, inpaint_config)
        model_names = [diffusion_model, text_encoder, vae]
        if run_unconditional_model:
            model_names.append(unconditional_model)
        if any(infer_model_format(name) == "gguf" for name in model_names):
            raise ValueError("Ideogram 4 does not currently support GGUF model files in this adapter.")
        warning = validation.validate_negative_prompt_policy(
            profile,
            negative_prompt,
            ignored_by_default=True,
        )
        validation.validate_gguf_available_for_models(
            gguf_backend.is_available(), *model_names
        )
        return [warning] if warning else []

    def generate(self, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress.phase("resolving models")
        return pipeline.generate_ideogram4_t2i(
            diffusion_model=kwargs["diffusion_model"],
            unconditional_model=kwargs["settings"].get("unconditional_model", ""),
            text_encoder=kwargs["text_encoder"],
            vae=kwargs["vae"],
            positive_prompt=kwargs["positive_prompt"],
            width=kwargs["width"],
            height=kwargs["height"],
            seed=kwargs["seed"],
            steps=int(kwargs["settings"]["steps"]),
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
