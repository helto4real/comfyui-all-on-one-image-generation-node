"""User-facing validation helpers for AIO image generation nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dimensions import parse_multiple_value
from .model_resolution import infer_model_format
from .profiles import ModelProfile
from .registry import list_model_types


def validate_model_type(model_type: str) -> None:
    if model_type not in list_model_types():
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported values: "
            f"{', '.join(list_model_types()) or 'none'}."
        )


def validate_settings_family(model_type: str, settings: Mapping[str, Any] | None) -> None:
    if not settings:
        return
    family = settings.get("family")
    if family != model_type:
        raise ValueError(
            f"Selected settings are for {family}, but model_type is {model_type}."
        )


def dimension_multiple_from_settings(
    settings: Mapping[str, Any],
    default_multiple: int,
) -> int | None:
    if "multiple_value" not in settings:
        return default_multiple
    return parse_multiple_value(settings.get("multiple_value"))


def validate_dimensions(width: int, height: int, multiple: int | None = 8) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if multiple is None:
        return
    if width % multiple != 0 or height % multiple != 0:
        raise ValueError(
            f"width and height must be multiples of {multiple} for this adapter."
        )


def validate_required_model_names(
    required_components: tuple[str, ...],
    diffusion_model: str,
    text_encoder: str,
    vae: str,
) -> None:
    values = {
        "diffusion_model": diffusion_model,
        "text_encoder": text_encoder,
        "vae": vae,
    }
    missing = [name for name in required_components if not values.get(name)]
    if missing:
        raise ValueError(f"Select required model files: {', '.join(missing)}.")


def validate_negative_prompt_policy(
    profile: ModelProfile,
    negative_prompt: str,
    *,
    ignored_by_default: bool = False,
) -> str | None:
    if not negative_prompt.strip():
        return None
    if profile.supports_negative_prompt:
        return None
    if ignored_by_default:
        return (
            f"{profile.display_name} profile does not use negative prompts by "
            "default; negative_prompt was ignored."
        )
    raise ValueError(f"{profile.display_name} does not currently support negative_prompt.")


def validate_reference_inputs(
    profile: ModelProfile,
    *,
    reference_image: Any = None,
    reference_inputs: Any = None,
    mask: Any = None,
) -> None:
    reference_count = 0
    if reference_image is not None:
        reference_count += 1
    if reference_inputs is not None:
        reference_count += int(getattr(reference_inputs, "count", 0))
        mask = mask if mask is not None else getattr(reference_inputs, "mask", None)

    if reference_count and not profile.supports_reference_image:
        raise ValueError(
            f"reference_image was connected, but {profile.key} currently supports "
            "text-to-image only in this adapter."
        )
    if mask is not None and not profile.supports_mask:
        raise ValueError(
            f"mask was connected, but {profile.key} currently supports text-to-image "
            "only in this adapter."
        )


def validate_inpaint_config(profile: ModelProfile, inpaint_config: Any = None) -> None:
    if inpaint_config is not None and not profile.supports_inpaint:
        raise ValueError(
            f"inpaint was connected, but {profile.key} does not currently support inpaint."
        )


def validate_gguf_available_for_models(
    available: bool,
    *model_names: str,
) -> None:
    if any(infer_model_format(name) == "gguf" for name in model_names) and not available:
        raise ValueError("A GGUF model file was selected, but no compatible GGUF backend was detected.")
