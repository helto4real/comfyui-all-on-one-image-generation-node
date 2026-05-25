"""Small helpers for classic nodes with dynamic optional inputs."""

from __future__ import annotations


class AnyType(str):
    """A ComfyUI wildcard input type for dynamic optional payloads."""

    def __ne__(self, value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """Dictionary that accepts any optional input key with the same type."""

    def __init__(self, input_type: str, data: dict | None = None):
        super().__init__()
        self.input_type = input_type
        self.data = data or {}
        self.update(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.input_type,)

    def __contains__(self, key):
        return True


ANY_TYPE = AnyType("*")
