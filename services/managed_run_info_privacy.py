"""Inactive shared projection for AIO run-info and private diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping

from helto_privacy import (
    SafeDiagnosticField,
    SafeDiagnosticKind,
    SensitiveFieldClass,
    SensitiveFieldDeclaration,
)

from .run_info import build_run_info_candidate


RUN_INFO_OPERATION_ID = "emit-run-info"
RUN_INFO_ADAPTER_ID = "run-info-projection"
RUN_INFO_RESOURCE_ID = "generate-prompts"

_BOOLEAN_FIELDS = (
    "configured",
    "fp16_accumulation_enabled",
    "resolved_fp16_accumulation_enabled",
    "memory_cleanup_applied",
    "duplicate_inpaint_reference_skipped",
)
_COUNT_FIELDS = ("duplicate_inpaint_reference_count",)


def run_info_sensitive_fields() -> tuple[SensitiveFieldDeclaration, ...]:
    return (
        SensitiveFieldDeclaration("*", SensitiveFieldClass.CONSUMER_DERIVED),
        SensitiveFieldDeclaration("debug", SensitiveFieldClass.DEBUG),
        SensitiveFieldDeclaration(
            "settings.positive_prompt_override",
            SensitiveFieldClass.USER_AUTHORED,
        ),
        SensitiveFieldDeclaration(
            "diffusion_model",
            SensitiveFieldClass.PATH_OR_NAME,
        ),
        SensitiveFieldDeclaration(
            "text_encoder",
            SensitiveFieldClass.PATH_OR_NAME,
        ),
        SensitiveFieldDeclaration("vae", SensitiveFieldClass.PATH_OR_NAME),
        SensitiveFieldDeclaration("loras", SensitiveFieldClass.PATH_OR_NAME),
    )


def run_info_safe_projection() -> tuple[SafeDiagnosticField, ...]:
    return (
        *tuple(
            SafeDiagnosticField(
                f"performance.{name}",
                SafeDiagnosticKind.BOOLEAN,
            )
            for name in _BOOLEAN_FIELDS
        ),
        *tuple(
            SafeDiagnosticField(
                f"performance.{name}",
                SafeDiagnosticKind.COUNT,
            )
            for name in _COUNT_FIELDS
        ),
        SafeDiagnosticField(
            "performance.warning_count",
            SafeDiagnosticKind.COUNT,
        ),
    )


class AioRunInfoProjectionAdapter:
    """Map product performance facts without deciding privacy policy."""

    def project(self, value: object, _declaration: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("AIO run-info must be an object.")
        performance = value.get("performance")
        if not isinstance(performance, Mapping):
            raise ValueError("AIO run-info performance facts are unavailable.")
        projected: dict[str, object] = {}
        for name in (*_BOOLEAN_FIELDS, *_COUNT_FIELDS):
            if name in performance:
                projected[name] = performance[name]
        warnings = performance.get("warnings")
        if warnings is None:
            projected["warning_count"] = 0
        elif isinstance(warnings, list):
            projected["warning_count"] = len(warnings)
        else:
            raise ValueError("AIO run-info warnings are invalid.")
        return {"performance": projected}


def project_managed_run_info(pack: object, run_info: object):
    """Project a built product value through server-resolved shared mode."""

    operations = getattr(pack, "operations", None)
    if not callable(operations):
        raise ValueError("Managed run-info privacy is unavailable.")
    return operations(RUN_INFO_RESOURCE_ID).project(RUN_INFO_OPERATION_ID, run_info)


def build_managed_run_info_json(pack: object, **facts: object) -> str:
    """Build the normal product schema, then apply the shared projection."""

    candidate = build_run_info_candidate(**facts)
    projected = project_managed_run_info(pack, candidate)
    return json.dumps(projected.value, indent=2, sort_keys=True)


__all__ = [
    "AioRunInfoProjectionAdapter",
    "RUN_INFO_ADAPTER_ID",
    "RUN_INFO_OPERATION_ID",
    "RUN_INFO_RESOURCE_ID",
    "build_managed_run_info_json",
    "project_managed_run_info",
    "run_info_safe_projection",
    "run_info_sensitive_fields",
]
