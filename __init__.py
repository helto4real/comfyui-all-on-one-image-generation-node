"""Classic ComfyUI node registration for AIO Image Generate."""

from __future__ import annotations

try:
    from .services import lora_info as _lora_info  # noqa: F401
    from .nodes import (
        AIOFlux2Klein9BSettings,
        AIOImageGenerate,
        AIOLoadPipelineModels,
        AIOLoraConfiguration,
        AIOZImageTurboSettings,
    )
except ImportError:  # pragma: no cover - direct pytest/importlib collection
    import services.lora_info as _lora_info  # noqa: F401
    from nodes import (
        AIOFlux2Klein9BSettings,
        AIOImageGenerate,
        AIOLoadPipelineModels,
        AIOLoraConfiguration,
        AIOZImageTurboSettings,
    )

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AIOImageGenerate": AIOImageGenerate,
    "AIOZImageTurboSettings": AIOZImageTurboSettings,
    "AIOFlux2Klein9BSettings": AIOFlux2Klein9BSettings,
    "AIOLoraConfiguration": AIOLoraConfiguration,
    "AIOLoadPipelineModels": AIOLoadPipelineModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIOImageGenerate": "AIO Image Generate",
    "AIOZImageTurboSettings": "Z-Image Turbo Settings",
    "AIOFlux2Klein9BSettings": "FLUX.2 Klein 9B Settings",
    "AIOLoraConfiguration": "AIO LoRA Configuration",
    "AIOLoadPipelineModels": "AIO Load Pipeline Models",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
