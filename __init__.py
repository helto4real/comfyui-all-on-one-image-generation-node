"""Classic ComfyUI node registration for AIO Image Generate."""

from __future__ import annotations

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

try:
    register_privacy_routes()
    register_ideogram4_prompt_library_routes()
    try:
        from helto_privacy import register_helto_privacy_ui

        register_helto_privacy_ui()
    except Exception:
        pass
except Exception:
    # Direct imports and tests often run outside ComfyUI's server process.
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
