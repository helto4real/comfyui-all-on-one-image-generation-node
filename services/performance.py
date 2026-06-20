"""Performance setting helpers for AIO model patching."""

from __future__ import annotations

import logging
from typing import Any, Callable


ATTENTION_MODES = ["auto", "off", "sage", "sage3", "flash", "xformers", "pytorch", "split", "sub_quad"]
TORCH_COMPILE_MODES = ["auto", "off", "on"]
TORCH_COMPILE_BACKENDS = ["inductor", "cudagraphs"]
PERFORMANCE_APPLY_TIMINGS = ["after_loras", "before_loras"]

DEFAULT_ATTENTION_MODE = "auto"
DEFAULT_TORCH_COMPILE_MODE = "off"
DEFAULT_TORCH_COMPILE_BACKEND = "inductor"
DEFAULT_PERFORMANCE_APPLY_TIMING = "after_loras"

INCOMPATIBLE_MASKED_ATTENTION_MODES = {"flash", "sage3"}


def performance_settings_present(settings: dict[str, Any]) -> bool:
    return any(
        key in settings
        for key in (
            "attention_mode",
            "torch_compile_mode",
            "torch_compile_backend",
            "performance_apply_timing",
        )
    )


def normalize_performance_apply_timing(settings: dict[str, Any]) -> str:
    timing = str(settings.get("performance_apply_timing", DEFAULT_PERFORMANCE_APPLY_TIMING))
    if timing not in PERFORMANCE_APPLY_TIMINGS:
        timing = DEFAULT_PERFORMANCE_APPLY_TIMING
    if performance_settings_present(settings):
        settings["performance_apply_timing"] = timing
    return timing


def apply_performance_settings(
    *,
    model: Any,
    settings: dict[str, Any],
    has_mask_or_reference: bool = False,
) -> Any:
    if not performance_settings_present(settings):
        return model

    warnings: list[str] = []
    attention_mode = str(settings.get("attention_mode", DEFAULT_ATTENTION_MODE))
    compile_mode = str(settings.get("torch_compile_mode", DEFAULT_TORCH_COMPILE_MODE))
    compile_backend = str(settings.get("torch_compile_backend", DEFAULT_TORCH_COMPILE_BACKEND))
    timing = normalize_performance_apply_timing(settings)

    attention_func, resolved_attention = _resolve_attention(
        attention_mode,
        model=model,
        has_mask_or_reference=has_mask_or_reference,
        warnings=warnings,
    )
    should_compile = _should_try_compile(compile_mode, warnings)
    compile_backend = compile_backend if compile_backend in TORCH_COMPILE_BACKENDS else DEFAULT_TORCH_COMPILE_BACKEND

    patched_model = model
    if attention_func is not None or should_compile:
        patched_model = _clone_model(model, disable_dynamic=should_compile, warnings=warnings)

    if attention_func is not None and not _apply_attention_override(
        patched_model,
        attention_func,
        resolved_attention,
        warnings,
    ):
        resolved_attention = "off"

    resolved_compile_mode = "off"
    resolved_compile_backend = "off"
    if should_compile:
        if _apply_torch_compile(patched_model, compile_backend, warnings):
            resolved_compile_mode = "on"
            resolved_compile_backend = compile_backend

    settings["attention_mode"] = attention_mode if attention_mode in ATTENTION_MODES else DEFAULT_ATTENTION_MODE
    settings["resolved_attention_mode"] = resolved_attention
    settings["torch_compile_mode"] = compile_mode if compile_mode in TORCH_COMPILE_MODES else DEFAULT_TORCH_COMPILE_MODE
    settings["torch_compile_backend"] = compile_backend
    settings["resolved_torch_compile_mode"] = resolved_compile_mode
    settings["resolved_torch_compile_backend"] = resolved_compile_backend
    settings["performance_apply_timing"] = timing
    if warnings:
        settings["performance_warnings"] = warnings
    return patched_model


def _resolve_attention(
    requested: str,
    *,
    model: Any,
    has_mask_or_reference: bool,
    warnings: list[str],
) -> tuple[Callable | None, str]:
    if requested not in ATTENTION_MODES:
        warnings.append(f"unknown attention mode '{requested}', leaving ComfyUI attention unchanged")
        return None, "off"
    if requested == "off":
        return None, "off"

    attention_module = _import_attention_module(warnings)
    if attention_module is None:
        return None, "off"

    if requested == "auto":
        for candidate in _auto_attention_priority(model, has_mask_or_reference):
            func = attention_module.get_attention_function(candidate, None)
            if func is not None:
                return func, candidate
        warnings.append("no compatible attention mode was available, leaving ComfyUI attention unchanged")
        return None, "off"

    if has_mask_or_reference and requested in INCOMPATIBLE_MASKED_ATTENTION_MODES:
        warnings.append(f"attention mode '{requested}' is not used with masks or reference inputs")
        return None, "off"

    func = attention_module.get_attention_function(requested, None)
    if func is None:
        warnings.append(f"attention mode '{requested}' is not installed, leaving ComfyUI attention unchanged")
        return None, "off"
    return func, requested


def _auto_attention_priority(model: Any, has_mask_or_reference: bool) -> list[str]:
    if _model_device_type(model) == "cpu":
        return ["sub_quad"]
    if has_mask_or_reference:
        return ["sage", "xformers", "pytorch", "split"]
    return ["sage3", "sage", "flash", "xformers", "pytorch", "split"]


def _model_device_type(model: Any) -> str:
    device = getattr(model, "load_device", None)
    if device is None:
        return ""
    return str(getattr(device, "type", device))


def _import_attention_module(warnings: list[str]):
    try:
        from comfy.ldm.modules import attention  # type: ignore

        return attention
    except Exception as exc:  # pragma: no cover - depends on ComfyUI runtime
        warnings.append(f"could not inspect ComfyUI attention backends: {exc}")
        logging.warning("[AIO Image Generate] Could not inspect ComfyUI attention backends: %s", exc)
        return None


def _should_try_compile(requested: str, warnings: list[str]) -> bool:
    if requested not in TORCH_COMPILE_MODES:
        warnings.append(f"unknown torch compile mode '{requested}', leaving compile disabled")
        return False
    if requested == "off":
        return False

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime
        warnings.append(f"could not import torch for compile support: {exc}")
        return False

    if not hasattr(torch, "compile"):
        warnings.append("torch.compile is not available")
        return False
    if requested == "auto" and not getattr(torch.cuda, "is_available", lambda: False)():
        return False
    return True


def _clone_model(model: Any, *, disable_dynamic: bool, warnings: list[str]) -> Any:
    clone = getattr(model, "clone", None)
    if clone is None:
        warnings.append("model could not be cloned before performance patching")
        return model
    try:
        return clone(disable_dynamic=disable_dynamic)
    except TypeError:
        return clone()


def _apply_attention_override(model: Any, attention_func: Callable, resolved_attention: str, warnings: list[str]) -> bool:
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        warnings.append(f"could not apply attention mode '{resolved_attention}' because model_options is unavailable")
        return False

    transformer_options = model_options.setdefault("transformer_options", {})

    def attention_override(_original_attention: Callable, *args: Any, **kwargs: Any):
        return attention_func(*args, **kwargs)

    transformer_options["optimized_attention_override"] = attention_override
    return True


def _apply_torch_compile(model: Any, backend: str, warnings: list[str]) -> bool:
    try:
        from comfy_api.torch_helpers import set_torch_compile_wrapper  # type: ignore

        set_torch_compile_wrapper(model=model, backend=backend, options={"guard_filter_fn": _skip_transformer_options_guard})
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime/backend
        warnings.append(f"could not enable torch compile with backend '{backend}': {exc}")
        logging.warning("[AIO Image Generate] Could not enable torch compile with backend '%s': %s", backend, exc)
        return False


def _skip_transformer_options_guard(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]
