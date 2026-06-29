"""Utility nodes for unpacking AIO generator info bundles."""

from __future__ import annotations

from typing import Any


def _info_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class AIOModelInfo:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE")
    RETURN_NAMES = ("model", "clip", "positive", "negative", "vae")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_info": (
                    "AIO_MODEL_INFO",
                    {"tooltip": "Model, CLIP, conditioning, and VAE values from AIO Image Generate."},
                ),
            },
            "hidden": {},
        }

    def extract(self, model_info: dict[str, Any] | None = None):
        info = _info_dict(model_info)
        return (
            info.get("model"),
            info.get("clip"),
            info.get("positive"),
            info.get("negative"),
            info.get("vae"),
        )


class AIOPIDInfo:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "sigma", "step")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_info": (
                    "AIO_PID_INFO",
                    {"tooltip": "PID capture values from AIO Image Generate."},
                ),
            },
            "hidden": {},
        }

    def extract(self, pid_info: dict[str, Any] | None = None):
        info = _info_dict(pid_info)
        return (
            info.get("latent"),
            float(info.get("sigma", 0.0) or 0.0),
            int(info.get("step", 0) or 0),
        )


class AIOInpaintInfo:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("source", "sample", "mask")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpaint_info": (
                    "AIO_INPAINT_INFO",
                    {"tooltip": "Inpaint debug previews from AIO Image Generate."},
                ),
            },
            "hidden": {},
        }

    def extract(self, inpaint_info: dict[str, Any] | None = None):
        info = _info_dict(inpaint_info)
        return (
            info.get("source"),
            info.get("sample"),
            info.get("mask"),
        )
