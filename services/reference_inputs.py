"""Reference image input normalization for image-edit adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REFERENCE_IMAGE_INPUT_NAMES = ("image 1", "image 2", "image 3", "image 4")


@dataclass(frozen=True)
class ReferenceInputs:
    images: tuple[Any, ...]
    mask: Any = None

    @property
    def count(self) -> int:
        return len(self.images)


def normalize_reference_inputs(
    values: Mapping[str, Any] | None = None,
    *,
    reference_image: Any = None,
    mask: Any = None,
) -> ReferenceInputs:
    """Collect ordered reference images from named ComfyUI sockets."""

    values = values or {}
    images: list[Any] = []
    first_gap: str | None = None
    for name in REFERENCE_IMAGE_INPUT_NAMES:
        image = values.get(name)
        if image is None:
            first_gap = first_gap or name
            continue
        if first_gap is not None:
            raise ValueError(
                f"{name} was connected, but {first_gap} is empty. Connect reference "
                "images in order from image 1."
            )
        images.append(image)

    if reference_image is not None:
        if images:
            raise ValueError("Use either image 1 or reference_image, not both.")
        images.append(reference_image)

    if mask is not None and not images:
        raise ValueError("mask can only be used when image 1 is connected.")

    return ReferenceInputs(images=tuple(images), mask=mask)
