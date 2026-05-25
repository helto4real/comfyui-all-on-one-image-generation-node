"""Output dimension helpers for AIO generation nodes."""

from __future__ import annotations

from dataclasses import dataclass


ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9")
SIZE_MODES = ("use aspect ratio", "use image 1 size")
SIZE_MODE_ASPECT_RATIO = "use aspect ratio"
SIZE_MODE_IMAGE_1 = "use image 1 size"
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_MAX_SIDE = 1024


@dataclass(frozen=True)
class ResolvedDimensions:
    width: int
    height: int
    max_side: int
    aspect_ratio: str
    size_mode: str


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}'. Supported values: "
            f"{', '.join(ASPECT_RATIOS)}."
        )
    width, height = aspect_ratio.split(":", 1)
    return int(width), int(height)


def infer_nearest_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return DEFAULT_ASPECT_RATIO

    actual = width / height
    return min(
        ASPECT_RATIOS,
        key=lambda ratio: abs((_parse_aspect_ratio(ratio)[0] / _parse_aspect_ratio(ratio)[1]) - actual),
    )


def resolve_dimensions_from_controls(
    *,
    size_mode: str | None = None,
    max_side: int | None,
    aspect_ratio: str | None,
    reference_inputs: object | None = None,
    legacy_width: int | None = None,
    legacy_height: int | None = None,
    default_width: int = DEFAULT_MAX_SIDE,
    default_height: int = DEFAULT_MAX_SIDE,
    dimension_multiple: int = 16,
) -> ResolvedDimensions:
    if max_side is None:
        if legacy_width is not None and legacy_height is not None:
            max_side = max(int(legacy_width), int(legacy_height))
            aspect_ratio = aspect_ratio or infer_nearest_aspect_ratio(
                int(legacy_width),
                int(legacy_height),
            )
        else:
            max_side = max(int(default_width), int(default_height))
            aspect_ratio = aspect_ratio or infer_nearest_aspect_ratio(
                int(default_width),
                int(default_height),
            )

    size_mode = size_mode or SIZE_MODE_ASPECT_RATIO
    if size_mode not in SIZE_MODES:
        raise ValueError(
            f"Unsupported size mode '{size_mode}'. Supported values: "
            f"{', '.join(SIZE_MODES)}."
        )

    if size_mode == SIZE_MODE_IMAGE_1:
        images = tuple(getattr(reference_inputs, "images", ()) or ())
        if not images:
            raise ValueError("size mode 'use image 1 size' requires image 1.")
        image = images[0]
        image_width = int(image.shape[2])
        image_height = int(image.shape[1])
        width = _round_to_multiple(float(image_width), dimension_multiple)
        height = _round_to_multiple(float(image_height), dimension_multiple)
        return ResolvedDimensions(
            width=width,
            height=height,
            max_side=max(width, height),
            aspect_ratio=infer_nearest_aspect_ratio(width, height),
            size_mode=size_mode,
        )

    aspect_ratio = aspect_ratio or DEFAULT_ASPECT_RATIO
    ratio_width, ratio_height = _parse_aspect_ratio(aspect_ratio)
    resolved_max_side = _round_to_multiple(float(max_side), dimension_multiple)

    if ratio_width >= ratio_height:
        width = resolved_max_side
        height = _round_to_multiple(
            resolved_max_side * ratio_height / ratio_width,
            dimension_multiple,
        )
    else:
        height = resolved_max_side
        width = _round_to_multiple(
            resolved_max_side * ratio_width / ratio_height,
            dimension_multiple,
        )

    return ResolvedDimensions(
        width=width,
        height=height,
        max_side=resolved_max_side,
        aspect_ratio=aspect_ratio,
        size_mode=size_mode,
    )
