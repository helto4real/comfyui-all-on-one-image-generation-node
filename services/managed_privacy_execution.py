"""Small AIO binding over the shared protected-execution capability."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Iterator


AIO_MANAGED_PRIVACY_PROFILE_ID = "helto.aio-image-generation"


def dispatch_aio_managed_execution(
    reference: object,
    execution_resource_id: str,
    context: Mapping[str, object],
    *,
    subject_id: object,
    cache_result: bool = True,
) -> object:
    """Resolve one injected reference and retain its result only in shared RAM."""

    if isinstance(reference, str):
        try:
            reference = json.loads(reference)
        except json.JSONDecodeError:
            raise ValueError("PRIVACY_EXECUTION_REFERENCE_INVALID") from None
    if not isinstance(reference, Mapping):
        raise ValueError("PRIVACY_EXECUTION_REFERENCE_INVALID")
    from helto_privacy.runtime import bound_privacy_pack

    execution = bound_privacy_pack(AIO_MANAGED_PRIVACY_PROFILE_ID).execution(
        execution_resource_id
    )
    result = execution.dispatch(reference, context, subject_id=subject_id)
    if cache_result:
        execution.cache_store(result.cache_identity, result.value)
    return result.value


def parse_aio_managed_reference(reference: object, error_code: str) -> Mapping[str, object]:
    if isinstance(reference, str):
        try:
            reference = json.loads(reference)
        except json.JSONDecodeError:
            raise ValueError(error_code) from None
    if not isinstance(reference, Mapping):
        raise ValueError(error_code)
    return reference


@contextmanager
def consume_aio_subject_mode(
    reference: object,
    binding_id: str,
    subject_id: object,
) -> Iterator[object]:
    """Consume one output-only subject reference for the exact Comfy node ID."""

    parsed = parse_aio_managed_reference(
        reference,
        "PRIVACY_SUBJECT_MODE_REFERENCE_INVALID",
    )
    from helto_privacy.runtime import bound_privacy_pack

    pack = bound_privacy_pack(AIO_MANAGED_PRIVACY_PROFILE_ID)
    with pack.subject_modes(binding_id).consume(parsed, subject_id) as lease:
        yield lease


def aio_subject_requires_private_execution(
    lease: object,
    binding_id: str,
) -> bool:
    """Validate an active AIO lease and return its server-attested mode."""

    from helto_privacy.runtime import bound_privacy_pack

    pack = bound_privacy_pack(AIO_MANAGED_PRIVACY_PROFILE_ID)
    check = getattr(lease, "requires_private_execution", None)
    if not callable(check):
        raise ValueError("PRIVACY_SUBJECT_MODE_REFERENCE_INVALID")
    return bool(check(profile=pack.profile, binding_id=binding_id))
