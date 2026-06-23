"""AIO model/CLIP loader node for external model patch chains."""

from __future__ import annotations

from typing import Any

try:
    from ..adapters import Flux2Klein9BAdapter, Ideogram4Adapter, Krea2Adapter, ZImageTurboAdapter  # noqa: F401
    from ..services import pipeline
    from ..services.lora_config import normalize_lora_config
    from ..services.registry import list_model_types
    from ..services.validation import validate_model_type, validate_settings_family
    from .aio_generate import _combined_filenames
except ImportError:  # pragma: no cover - direct test imports
    from adapters import Flux2Klein9BAdapter, Ideogram4Adapter, Krea2Adapter, ZImageTurboAdapter  # noqa: F401
    from services import pipeline
    from services.lora_config import normalize_lora_config
    from services.registry import list_model_types
    from services.validation import validate_model_type, validate_settings_family
    from nodes.aio_generate import _combined_filenames


class AIOLoadPipelineModels:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    list_model_types(),
                    {"tooltip": "Select the model family/profile used to load a compatible text encoder type."},
                ),
                "diffusion_model": (
                    _combined_filenames(
                        (
                            "diffusion_models",
                            "unet",
                            "checkpoints",
                            "unet_gguf",
                            "model_gguf",
                        )
                    ),
                    {"tooltip": "Diffusion model file to load before optional external model patch nodes."},
                ),
                "text_encoder": (
                    _combined_filenames(("text_encoders", "clip", "clip_gguf")),
                    {"tooltip": "Text encoder or CLIP file to load before optional external CLIP patch nodes."},
                ),
            },
            "optional": {
                "model_settings": (
                    "AIO_MODEL_SETTINGS",
                    {"tooltip": "Optional matching settings object used for precision policy and model performance patches."},
                ),
                "lora_config": (
                    "AIO_LORA_CONFIG",
                    {"tooltip": "Optional AIO LoRA stack to apply before the model and CLIP are output."},
                ),
            },
            "hidden": {},
        }

    def load(
        self,
        model_type: str,
        diffusion_model: str,
        text_encoder: str,
        model_settings: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
    ):
        validate_model_type(model_type)
        validate_settings_family(model_type, model_settings)
        settings = dict(model_settings or {})
        normalized_lora_config = normalize_lora_config(lora_config)
        model = pipeline.load_diffusion_model(
            diffusion_model=diffusion_model,
            precision_policy=settings.get("precision_policy"),
        )
        clip = pipeline.load_text_encoder(
            text_encoder=text_encoder,
            clip_type=pipeline.text_encoder_clip_type(model_type),
        )
        apply_timing = pipeline.normalize_performance_apply_timing(settings)
        if apply_timing == "before_loras":
            model = pipeline.apply_model_performance(model=model, settings=settings)
        if model_type == "ideogram4":
            model, _ = pipeline.apply_lora_config_model_only(
                model=model,
                lora_config=normalized_lora_config,
            )
        else:
            model, clip, _ = pipeline.apply_lora_config(
                model=model,
                clip=clip,
                lora_config=normalized_lora_config,
            )
        if apply_timing == "after_loras":
            model = pipeline.apply_model_performance(model=model, settings=settings)
        return model, clip
