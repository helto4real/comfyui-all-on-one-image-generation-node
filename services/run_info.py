"""Run-info helpers for JSON output from the AIO facade node."""

from __future__ import annotations

import json
from typing import Any


PERFORMANCE_KEYS = (
    "attention_mode",
    "resolved_attention_mode",
    "torch_compile_mode",
    "torch_compile_backend",
    "resolved_torch_compile_mode",
    "resolved_torch_compile_backend",
    "performance_apply_timing",
    "performance_warnings",
)


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
) -> dict[str, Any]:
    return {
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
        "settings": settings,
        "performance": performance_info_from_settings(settings),
        "warnings": warnings,
        "adapter_version": adapter_version,
        "loras": loras or [],
    }


def to_json(run_info: dict[str, Any]) -> str:
    return json.dumps(run_info, indent=2, sort_keys=True)
