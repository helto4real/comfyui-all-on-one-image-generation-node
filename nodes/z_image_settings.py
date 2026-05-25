"""Classic ComfyUI settings node for Z-Image Turbo."""

from __future__ import annotations


class AIOZImageTurboSettings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speed_preset": (["default", "quality", "experimental"],),
                "force_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "prompt_enhance": (["off", "light", "strong"],),
                "ignore_negative_prompt": ("BOOLEAN", {"default": True}),
                "precision_policy": (["auto", "fp8", "bf16"],),
            }
        }

    def build_settings(
        self,
        speed_preset: str,
        force_steps: int,
        prompt_enhance: str,
        ignore_negative_prompt: bool,
        precision_policy: str,
    ):
        return (
            {
                "family": "z_image_turbo",
                "speed_preset": speed_preset,
                "force_steps": force_steps,
                "prompt_enhance": prompt_enhance,
                "ignore_negative_prompt": ignore_negative_prompt,
                "precision_policy": precision_policy,
            },
        )
