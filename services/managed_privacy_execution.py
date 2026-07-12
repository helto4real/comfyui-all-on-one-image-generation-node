"""Small AIO binding over the shared protected-execution capability."""

from __future__ import annotations

import json
from collections.abc import Mapping


AIO_MANAGED_PRIVACY_PROFILE_ID = "helto.aio-image-generation"


def dispatch_aio_managed_execution(
    reference: object,
    execution_resource_id: str,
    context: Mapping[str, object],
    *,
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
    result = execution.dispatch(reference, context)
    if cache_result:
        execution.cache_store(result.cache_identity, result.value)
    return result.value
