"""Model-family profile definitions for the AIO image generation facade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    family: str
    default_steps: int
    default_cfg: float
    default_width: int
    default_height: int
    default_sampler: str
    default_scheduler: str
    supports_negative_prompt: bool
    supports_reference_image: bool
    supports_mask: bool
    supports_gguf: bool
    required_components: tuple[str, ...]
    notes: str


PROFILES: dict[str, ModelProfile] = {
    "z_image_turbo": ModelProfile(
        key="z_image_turbo",
        display_name="Z-Image Turbo",
        family="z_image_turbo",
        default_steps=8,
        default_cfg=1.0,
        default_width=1024,
        default_height=1024,
        default_sampler="auto",
        default_scheduler="auto",
        supports_negative_prompt=False,
        supports_reference_image=False,
        supports_mask=False,
        supports_gguf=True,
        required_components=("diffusion_model", "text_encoder", "vae"),
        notes=(
            "Text-to-image scaffold for Z-Image Turbo. Negative prompts are "
            "ignored by default."
        ),
    ),
    "flux2_klein_9b": ModelProfile(
        key="flux2_klein_9b",
        display_name="FLUX.2 Klein 9B",
        family="flux2_klein_9b",
        default_steps=4,
        default_cfg=1.0,
        default_width=1024,
        default_height=1024,
        default_sampler="auto",
        default_scheduler="auto",
        supports_negative_prompt=True,
        supports_reference_image=False,
        supports_mask=False,
        supports_gguf=True,
        required_components=("diffusion_model", "text_encoder", "vae"),
        notes=(
            "Distilled text-to-image defaults. Reference editing settings are "
            "accepted by the settings node but are not implemented yet."
        ),
    ),
}


def list_profiles() -> tuple[ModelProfile, ...]:
    return tuple(PROFILES.values())


def get_profile(profile_key: str) -> ModelProfile:
    try:
        return PROFILES[profile_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type '{profile_key}'.") from exc
