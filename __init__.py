"""Classic ComfyUI node registration for AIO Image Generate."""

from __future__ import annotations

import logging
import sys

try:
    from .services import lora_info as _lora_info  # noqa: F401
    from .routes.ideogram4_prompt_library import register_ideogram4_prompt_library_routes
    from .routes.privacy import register_privacy_routes
    from .nodes import (
        AIOFlux2Klein9BSettings,
        AIOIdeogram4PromptBuilder,
        AIOIdeogram4Settings,
        AIOInpaintInfo,
        AIOImageGenerate,
        AIOInpaint,
        AIOKrea2Settings,
        AIOModelInfo,
        AIOPIDInfo,
        AIOLoadPipelineModels,
        AIOLoraConfiguration,
        AIOZImageTurboSettings,
    )
except ImportError:  # pragma: no cover - direct pytest/importlib collection
    import services.lora_info as _lora_info  # noqa: F401
    from routes.ideogram4_prompt_library import register_ideogram4_prompt_library_routes
    from routes.privacy import register_privacy_routes
    from nodes import (
        AIOFlux2Klein9BSettings,
        AIOIdeogram4PromptBuilder,
        AIOIdeogram4Settings,
        AIOInpaintInfo,
        AIOImageGenerate,
        AIOInpaint,
        AIOKrea2Settings,
        AIOModelInfo,
        AIOPIDInfo,
        AIOLoadPipelineModels,
        AIOLoraConfiguration,
        AIOZImageTurboSettings,
    )

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AIOImageGenerate": AIOImageGenerate,
    "AIOZImageTurboSettings": AIOZImageTurboSettings,
    "AIOFlux2Klein9BSettings": AIOFlux2Klein9BSettings,
    "AIOIdeogram4PromptBuilder": AIOIdeogram4PromptBuilder,
    "AIOIdeogram4Settings": AIOIdeogram4Settings,
    "AIOKrea2Settings": AIOKrea2Settings,
    "AIOInpaint": AIOInpaint,
    "AIOLoraConfiguration": AIOLoraConfiguration,
    "AIOLoadPipelineModels": AIOLoadPipelineModels,
    "AIOModelInfo": AIOModelInfo,
    "AIOPIDInfo": AIOPIDInfo,
    "AIOInpaintInfo": AIOInpaintInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIOImageGenerate": "AIO Image Generate",
    "AIOZImageTurboSettings": "Z-Image Turbo Settings",
    "AIOFlux2Klein9BSettings": "FLUX.2 Klein 9B Settings",
    "AIOIdeogram4PromptBuilder": "Ideogram 4 Prompt Builder",
    "AIOIdeogram4Settings": "Ideogram 4 Settings",
    "AIOKrea2Settings": "Krea 2 Settings",
    "AIOInpaint": "AIO Inpaint",
    "AIOLoraConfiguration": "AIO LoRA Configuration",
    "AIOLoadPipelineModels": "AIO Load Pipeline Models",
    "AIOModelInfo": "AIO Model Info",
    "AIOPIDInfo": "AIO PID Info",
    "AIOInpaintInfo": "AIO Inpaint Info",
}


def _register_safely(label, callback) -> None:
    try:
        callback()
    except Exception as exc:  # noqa: BLE001 - one optional route family must not block another.
        logging.warning("AIO Image Generate could not register %s: %s", label, exc)


def _register_shared_privacy_ui() -> None:
    from helto_privacy import register_helto_privacy_ui

    server_module = sys.modules.get("server")
    prompt_server = getattr(getattr(server_module, "PromptServer", None), "instance", None)
    if prompt_server is not None:
        register_helto_privacy_ui(prompt_server=prompt_server)


_register_safely("LoRA information routes", _lora_info.register_routes)
_register_safely("privacy routes", register_privacy_routes)
_register_safely("Ideogram prompt-library routes", register_ideogram4_prompt_library_routes)
_register_safely("shared privacy UI routes", _register_shared_privacy_ui)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
