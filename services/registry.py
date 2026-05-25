"""Adapter registry for model-family implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .profiles import ModelProfile, get_profile as get_declared_profile

if TYPE_CHECKING:
    from adapters.base import BaseImageAdapter


_ADAPTERS: dict[str, type["BaseImageAdapter"]] = {}


def register_adapter(adapter_cls: type["BaseImageAdapter"]) -> type["BaseImageAdapter"]:
    model_type = adapter_cls.model_type
    if not model_type:
        raise ValueError("Adapter classes must define a non-empty model_type.")
    _ADAPTERS[model_type] = adapter_cls
    return adapter_cls


def list_model_types() -> list[str]:
    return sorted(_ADAPTERS)


def get_profile_for_model(model_type: str) -> ModelProfile:
    if model_type not in _ADAPTERS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported values: "
            f"{', '.join(list_model_types()) or 'none'}."
        )
    return get_declared_profile(_ADAPTERS[model_type].profile_key)


def get_profile(model_type: str) -> ModelProfile:
    return get_profile_for_model(model_type)


def get_adapter(model_type: str) -> "BaseImageAdapter":
    try:
        adapter_cls = _ADAPTERS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported values: "
            f"{', '.join(list_model_types()) or 'none'}."
        ) from exc
    return adapter_cls()


def clear_registry_for_tests() -> None:
    _ADAPTERS.clear()
