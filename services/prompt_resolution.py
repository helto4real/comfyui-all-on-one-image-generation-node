"""Crypto-free Generate/Krea prompt resolution shared by both cutover paths."""

from __future__ import annotations

from collections.abc import Mapping


GENERATE_EXECUTION_RESOURCE_ID = "generate-execution"
KREA_EXECUTION_RESOURCE_ID = "krea-inpaint-execution"
MASKED_PROMPT_VALUE = "Private prompt - hover to reveal"
_MASKED_PROMPT_VALUES = frozenset({"[private prompt]", MASKED_PROMPT_VALUE})


def resolve_prompt_source(
    value: object,
    *,
    linked: bool,
    workflow_value: object = None,
) -> str:
    """Preserve ComfyUI's linked-input and unlinked workflow fallback rules."""

    resolved = normalize_prompt_text(value)
    if linked or resolved.strip():
        return resolved
    if workflow_value is None or workflow_value in _MASKED_PROMPT_VALUES:
        return resolved
    fallback = normalize_prompt_text(workflow_value)
    return fallback if fallback.strip() else resolved


def prompt_input_is_link(prompt: object, unique_id: object, input_name: str) -> bool:
    if not isinstance(prompt, Mapping) or unique_id is None:
        return False
    node = prompt.get(str(unique_id), prompt.get(unique_id))
    if not isinstance(node, Mapping):
        return False
    inputs = node.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return False
    value = inputs.get(input_name)
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
    )


def normalize_prompt_text(value: object) -> str:
    if isinstance(value, Mapping) and set(value) == {"value"}:
        value = value["value"]
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("AIO prompt text is invalid.")
    return value
