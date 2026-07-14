"""Classic ComfyUI node registration for AIO Image Generate."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from .services import lora_info as _lora_info  # noqa: F401
    from .services.managed_prompt_privacy import install_aio_privacy
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
    from services.managed_prompt_privacy import install_aio_privacy
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
_PACKAGE_ROOT = Path(__file__).resolve().parent

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
    register_helto_privacy_ui(
        legacy_key_dir=_PACKAGE_ROOT / "config",
        prompt_server=prompt_server,
    )


_register_safely("LoRA information routes", _lora_info.register_routes)
_register_shared_privacy_ui()
install_aio_privacy(_PACKAGE_ROOT)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
