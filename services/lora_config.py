"""LoRA configuration normalization and validation."""

from __future__ import annotations

import os
import re
from typing import Any

LORA_CONFIG_VERSION = 1


def _available_loras() -> list[str] | None:
    try:
        import folder_paths  # type: ignore

        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return None


def _resolve_lora_name(name: str, available_loras: list[str] | None = None) -> str:
    lora_paths = available_loras if available_loras is not None else _available_loras()
    if lora_paths is None:
        return name

    if name in lora_paths:
        return name

    lora_paths_no_ext = [os.path.splitext(path)[0] for path in lora_paths]
    if name in lora_paths_no_ext:
        return lora_paths[lora_paths_no_ext.index(name)]

    name_no_ext = os.path.splitext(name)[0]
    if name_no_ext in lora_paths_no_ext:
        return lora_paths[lora_paths_no_ext.index(name_no_ext)]

    lora_filenames = [os.path.basename(path) for path in lora_paths]
    if name in lora_filenames:
        return lora_paths[lora_filenames.index(name)]

    name_filename = os.path.basename(name)
    if name_filename in lora_filenames:
        return lora_paths[lora_filenames.index(name_filename)]

    lora_filenames_no_ext = [os.path.splitext(filename)[0] for filename in lora_filenames]
    if name in lora_filenames_no_ext:
        return lora_paths[lora_filenames_no_ext.index(name)]

    name_filename_no_ext = os.path.splitext(name_filename)[0]
    if name_filename_no_ext in lora_filenames_no_ext:
        return lora_paths[lora_filenames_no_ext.index(name_filename_no_ext)]

    for lora_path in lora_paths:
        if name in lora_path:
            return lora_path

    raise ValueError(f"LoRA '{name}' was selected, but it was not found in ComfyUI's loras folder.")


def _sort_lora_key(item: tuple[str, Any]) -> tuple[int, int | str]:
    key, _ = item
    match = re.match(r"^lora[_-]?(\d+)$", key.lower())
    if match:
        return 0, int(match.group(1))
    return 1, key


def _iter_raw_loras(payload: dict[str, Any]):
    if isinstance(payload.get("loras"), list):
        yield from payload["loras"]
        return

    for key, value in sorted(payload.items(), key=_sort_lora_key):
        if key.lower().startswith("lora_") and isinstance(value, dict):
            yield value


def normalize_lora_config(
    payload: dict[str, Any] | None,
    *,
    available_loras: list[str] | None = None,
) -> dict[str, Any]:
    """Return a stable, executable LoRA config dict."""

    payload = payload or {}
    ui = payload.get("ui") if isinstance(payload.get("ui"), dict) else {}
    show_strengths = str(
        payload.get("show_strengths")
        or ui.get("show_strengths")
        or payload.get("Show Strengths")
        or "single"
    )
    match = str(payload.get("match") or ui.get("match") or payload.get("Match") or "")

    normalized: list[dict[str, Any]] = []
    for raw_lora in _iter_raw_loras(payload):
        if not isinstance(raw_lora, dict):
            continue
        enabled = bool(raw_lora.get("enabled", raw_lora.get("on", True)))
        name = raw_lora.get("name", raw_lora.get("lora"))
        if not enabled or not name or name == "None":
            continue

        strength_model = float(raw_lora.get("strength_model", raw_lora.get("strength", 1.0)))
        strength_clip_raw = raw_lora.get("strength_clip", raw_lora.get("strengthTwo"))
        strength_clip = strength_model if strength_clip_raw is None else float(strength_clip_raw)
        if strength_model == 0 and strength_clip == 0:
            continue

        normalized.append(
            {
                "enabled": True,
                "name": _resolve_lora_name(str(name), available_loras),
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            }
        )

    return {
        "version": LORA_CONFIG_VERSION,
        "loras": normalized,
        "ui": {
            "show_strengths": show_strengths,
            "match": match,
        },
    }


def summarize_loras(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    normalized = normalize_lora_config(config)
    return [dict(row) for row in normalized["loras"]]
