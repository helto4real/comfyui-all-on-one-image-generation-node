"""Run-info helpers for JSON output from the AIO facade node."""

from __future__ import annotations

import json
from typing import Any

from . import privacy


PERFORMANCE_KEYS = (
    "attention_mode",
    "resolved_attention_mode",
    "torch_compile_mode",
    "torch_compile_backend",
    "resolved_torch_compile_mode",
    "resolved_torch_compile_backend",
    "performance_apply_timing",
    "fp16_accumulation_enabled",
    "resolved_fp16_accumulation_enabled",
    "memory_policy",
    "resolved_memory_policy",
    "memory_cleanup_applied",
    "memory_reserved_vram_gb",
    "duplicate_inpaint_reference_skipped",
    "duplicate_inpaint_reference_count",
    "performance_warnings",
)
SENSITIVE_SETTINGS_KEYS = (
    "positive_prompt_override",
)


def settings_info_from_settings(settings: dict[str, Any], privacy_mode: bool = False) -> dict[str, Any]:
    info = dict(settings)
    if not privacy_mode:
        return info
    for key in SENSITIVE_SETTINGS_KEYS:
        value = info.get(key)
        if value in (None, "") or privacy.is_encrypted_payload(value):
            continue
        info[key] = privacy.encrypt_state({"value": str(value)})
    return info


def performance_info_from_settings(settings: dict[str, Any]) -> dict[str, Any]:
    configured = any(key in settings for key in PERFORMANCE_KEYS)
    info = {
        "configured": configured,
        "attention_mode": settings.get("attention_mode", "off"),
        "resolved_attention_mode": settings.get("resolved_attention_mode", "off"),
        "torch_compile_mode": settings.get("torch_compile_mode", "off"),
        "torch_compile_backend": settings.get("torch_compile_backend", "inductor"),
        "resolved_torch_compile_mode": settings.get("resolved_torch_compile_mode", "off"),
        "resolved_torch_compile_backend": settings.get("resolved_torch_compile_backend", "off"),
        "performance_apply_timing": settings.get("performance_apply_timing", "after_loras"),
    }
    if "performance_warnings" in settings:
        info["warnings"] = settings["performance_warnings"]
    if "fp16_accumulation_enabled" in settings or "resolved_fp16_accumulation_enabled" in settings:
        info["fp16_accumulation_enabled"] = bool(settings.get("fp16_accumulation_enabled", False))
        info["resolved_fp16_accumulation_enabled"] = bool(settings.get("resolved_fp16_accumulation_enabled", False))
    if "memory_policy" in settings or "resolved_memory_policy" in settings:
        info["memory_policy"] = settings.get("memory_policy", "balanced")
        info["resolved_memory_policy"] = settings.get("resolved_memory_policy", info["memory_policy"])
        info["memory_cleanup_applied"] = bool(settings.get("memory_cleanup_applied", False))
        info["memory_reserved_vram_gb"] = float(settings.get("memory_reserved_vram_gb", 0.0))
    if settings.get("duplicate_inpaint_reference_skipped"):
        info["duplicate_inpaint_reference_skipped"] = True
        info["duplicate_inpaint_reference_count"] = int(settings.get("duplicate_inpaint_reference_count", 0))
    return info


def build_run_info(
    *,
    model_type: str,
    display_name: str,
    diffusion_model: str,
    diffusion_model_format: str,
    text_encoder: str,
    text_encoder_format: str,
    vae: str,
    vae_format: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    warnings: list[str],
    adapter_version: str,
    loras: list[dict[str, Any]] | None = None,
    privacy_mode: bool = False,
    debug: dict[str, Any] | None = None,
    second_pass: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings_info = settings_info_from_settings(settings, privacy_mode=privacy_mode)
    info = {
        "model_type": model_type,
        "display_name": display_name,
        "diffusion_model": diffusion_model,
        "diffusion_model_format": diffusion_model_format,
        "text_encoder": text_encoder,
        "text_encoder_format": text_encoder_format,
        "vae": vae,
        "vae_format": vae_format,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "settings": settings_info,
        "performance": performance_info_from_settings(settings_info),
        "warnings": warnings,
        "adapter_version": adapter_version,
        "loras": loras or [],
        "second_pass": second_pass or {"enabled": False, "applied": False},
    }
    if debug is not None:
        info["debug"] = debug
    return info


def to_json(run_info: dict[str, Any]) -> str:
    return json.dumps(run_info, indent=2, sort_keys=True)
