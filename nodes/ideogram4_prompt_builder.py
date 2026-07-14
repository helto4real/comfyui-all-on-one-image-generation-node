"""Classic ComfyUI node for building Ideogram 4 structured prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

try:
    from ..services.dimensions import (
        ASPECT_RATIOS,
        DEFAULT_ASPECT_RATIO,
        DEFAULT_MAX_SIDE,
        DEFAULT_MULTIPLE_VALUE,
        MULTIPLE_VALUES,
        SIZE_MODE_ASPECT_RATIO,
        resolve_dimensions_from_controls,
    )
    from ..services import ideogram4_prompt_builder as prompt_builder
    from ..services.managed_builder_privacy import BUILDER_SUBJECT_MODE_BINDING_ID
    from ..services.managed_privacy_execution import (
        aio_subject_requires_private_execution,
        consume_aio_subject_mode,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services.dimensions import (
        ASPECT_RATIOS,
        DEFAULT_ASPECT_RATIO,
        DEFAULT_MAX_SIDE,
        DEFAULT_MULTIPLE_VALUE,
        MULTIPLE_VALUES,
        SIZE_MODE_ASPECT_RATIO,
        resolve_dimensions_from_controls,
    )
    from services import ideogram4_prompt_builder as prompt_builder
    from services.managed_builder_privacy import BUILDER_SUBJECT_MODE_BINDING_ID
    from services.managed_privacy_execution import (
        aio_subject_requires_private_execution,
        consume_aio_subject_mode,
    )


STYLE_OPTIONS = ("none", "photo", "art_style")
IMPORT_MODES = ("when empty", "always")
OUTPUT_FORMATS = ("compact", "pretty")
COORD_MODES = ("normalized", "absolute")
BBOX_ORDERS = ("yx", "xy")
_MANAGED_EXECUTION_CAPABILITY = object()


def _external_cache_providers_registered() -> bool:
    try:
        from comfy_execution.cache_provider import _has_cache_providers  # type: ignore

        return bool(_has_cache_providers())
    except Exception:
        return False


def _plain_builder_text(value: object) -> str:
    if isinstance(value, Mapping):
        raise ValueError("Protected builder fields require managed private execution.")
    return "" if value is None else str(value)


class AIOIdeogram4PromptBuilder:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_IDEOGRAM4_PROMPT", "STRING", "IMAGE", "BOUNDING_BOX", "INT", "INT")
    RETURN_NAMES = ("prompt_builder", "prompt", "preview", "bboxes", "width", "height")
    FUNCTION = "build_prompt"

    @classmethod
    def IS_CHANGED(cls, privacy_mode: bool = False, **kwargs):
        del cls, kwargs
        if bool(privacy_mode) and _external_cache_providers_registered():
            return float("NaN")
        return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max side": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_SIDE,
                        "min": 256,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Longest generated edge in pixels. Uses the same calculation as AIO Image Generate.",
                    },
                ),
                "aspect ratio": (
                    list(ASPECT_RATIOS),
                    {
                        "default": DEFAULT_ASPECT_RATIO,
                        "tooltip": "Output shape to use with max side. Uses the same calculation as AIO Image Generate.",
                    },
                ),
                "multiple value": (
                    list(MULTIPLE_VALUES),
                    {
                        "default": DEFAULT_MULTIPLE_VALUE,
                        "tooltip": "Round dimensions to the selected multiple. Ideogram 4 should normally use 16.",
                    },
                ),
                "privacy_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Encrypt prompt-builder text and editor state in saved workflows and hide it unless hovered.",
                    },
                ),
                "high_level_description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional one-line overview of the whole image. Blank is omitted.",
                    },
                ),
                "background": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Scene background description.",
                    },
                ),
                "style": (
                    list(STYLE_OPTIONS),
                    {"default": "photo", "tooltip": "Ideogram structured style block type."},
                ),
                "photo": (
                    "STRING",
                    {"default": "", "tooltip": "Photo style descriptor used when style is photo."},
                ),
                "art_style": (
                    "STRING",
                    {"default": "", "tooltip": "Art style descriptor used when style is art_style."},
                ),
                "aesthetics": (
                    "STRING",
                    {"default": "", "tooltip": "Style descriptor emitted when style is not none."},
                ),
                "lighting": (
                    "STRING",
                    {"default": "", "tooltip": "Style descriptor emitted when style is not none."},
                ),
                "medium": (
                    "STRING",
                    {"default": "", "tooltip": "Style descriptor emitted when style is not none."},
                ),
                "import_mode": (
                    list(IMPORT_MODES),
                    {
                        "default": "when empty",
                        "tooltip": "Use wired import_json always, or only to seed an empty editor.",
                    },
                ),
                "output_format": (
                    list(OUTPUT_FORMATS),
                    {
                        "default": "compact",
                        "tooltip": "Compact matches the Ideogram 4 training format; pretty is for readability.",
                    },
                ),
                "style_palette_data": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Serialized style color palette managed by the node UI.",
                    },
                ),
                "elements_data": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Serialized region data managed by the node UI.",
                    },
                ),
                "bg_brightness": (
                    "INT",
                    {
                        "default": 25,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Background image brightness for the preview/editor.",
                    },
                ),
                "coord_mode": (
                    list(COORD_MODES),
                    {
                        "default": "normalized",
                        "tooltip": "BBox coordinate scale for JSON output: normalized 0-1000 grid or absolute pixels.",
                    },
                ),
                "bbox_order": (
                    list(BBOX_ORDERS),
                    {
                        "default": "yx",
                        "tooltip": "BBox axis order for JSON output: yx is Ideogram [ymin,xmin,ymax,xmax], xy is Qwen-style [xmin,ymin,xmax,ymax].",
                    },
                ),
            },
            "optional": {
                "privacy_mode_reference": (
                    "STRING",
                    {
                        "default": "",
                        "socketless": True,
                        "hidden": True,
                        "tooltip": "Managed subject privacy-mode reference injected by the shared privacy barrier.",
                    },
                ),
                "private_execution": (
                    "STRING",
                    {
                        "default": "",
                        "socketless": True,
                        "hidden": True,
                        "tooltip": "Managed private builder execution reference injected by the shared privacy barrier.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Optional reference image shown behind the preview/editor regions."},
                ),
                "import_json": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Optional full caption JSON to import into the editor/output.",
                    },
                ),
                "bboxes": (
                    "BOUNDING_BOX",
                    {
                        "forceInput": True,
                        "tooltip": "Optional pixel-space boxes used to seed editor regions when it has none.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def build_prompt(
        self,
        high_level_description: str = "",
        background: str = "",
        style: str = "photo",
        photo: str = "",
        art_style: str = "",
        aesthetics: str = "",
        lighting: str = "",
        medium: str = "",
        import_mode: str = "when empty",
        output_format: str = "compact",
        coord_mode: str = "normalized",
        bbox_order: str = "yx",
        style_palette_data: str = "",
        elements_data: str = "",
        bg_brightness: int = 25,
        image: Any = None,
        import_json: str = "",
        bboxes: Any = None,
        privacy_mode: bool = False,
        privacy_mode_reference: str = "",
        private_execution: str = "",
        unique_id: str | None = None,
        _subject_mode_lease: object = None,
        _managed_execution_capability: object = None,
        **dimension_values: Any,
    ):
        if _subject_mode_lease is None and privacy_mode_reference:
            inputs = dict(locals())
            inputs.pop("self")
            inputs.pop("dimension_values")
            inputs.pop("_subject_mode_lease")
            inputs.pop("_managed_execution_capability")
            inputs.update(dimension_values)
            with consume_aio_subject_mode(
                privacy_mode_reference,
                BUILDER_SUBJECT_MODE_BINDING_ID,
                unique_id,
            ) as lease:
                inputs["_subject_mode_lease"] = lease
                return self.build_prompt(**inputs)
        if (
            _subject_mode_lease is not None
            and aio_subject_requires_private_execution(
                _subject_mode_lease,
                BUILDER_SUBJECT_MODE_BINDING_ID,
            )
            and not private_execution
            and _managed_execution_capability is not _MANAGED_EXECUTION_CAPABILITY
        ):
            raise ValueError(
                "Private prompt builder requires a managed execution reference."
            )
        if _subject_mode_lease is None and (
            private_execution or unique_id is not None or privacy_mode
        ):
            raise ValueError(
                "Prompt builder requires managed references for subject-mode and execution."
            )
        if private_execution:
            try:
                from ..services.managed_builder_privacy import dispatch_aio_builder_execution
            except ImportError:  # pragma: no cover - direct test imports
                from services.managed_builder_privacy import dispatch_aio_builder_execution

            product_inputs = {
                "high_level_description": high_level_description,
                "background": background,
                "style": style,
                "photo": photo,
                "art_style": art_style,
                "aesthetics": aesthetics,
                "lighting": lighting,
                "medium": medium,
                "import_mode": import_mode,
                "output_format": output_format,
                "coord_mode": coord_mode,
                "bbox_order": bbox_order,
                "style_palette_data": style_palette_data,
                "elements_data": elements_data,
                "bg_brightness": bg_brightness,
                "image": image,
                "import_json": import_json,
                "bboxes": bboxes,
                "privacy_mode": privacy_mode,
                "unique_id": unique_id,
                "_subject_mode_lease": _subject_mode_lease,
                "_managed_execution_capability": _MANAGED_EXECUTION_CAPABILITY,
                **dimension_values,
            }

            def build_resolved_prompt(semantic: object):
                if not isinstance(semantic, dict) or not isinstance(semantic.get("widgets"), dict):
                    raise ValueError("AIO builder execution state is invalid.")
                resolved_inputs = dict(product_inputs)
                resolved_inputs.update(semantic["widgets"])
                effective_mode = semantic.get("effective_privacy_mode")
                if not isinstance(effective_mode, bool):
                    raise ValueError("AIO builder effective privacy mode is invalid.")
                resolved_inputs["privacy_mode"] = effective_mode
                return self.build_prompt(**resolved_inputs)

            return dispatch_aio_builder_execution(
                private_execution,
                {"dispatch": build_resolved_prompt},
                subject_id=unique_id,
            )

        high_level_description = _plain_builder_text(high_level_description)
        background = _plain_builder_text(background)
        photo = _plain_builder_text(photo)
        art_style = _plain_builder_text(art_style)
        aesthetics = _plain_builder_text(aesthetics)
        lighting = _plain_builder_text(lighting)
        medium = _plain_builder_text(medium)
        style_palette_data = _plain_builder_text(style_palette_data)
        elements_data = _plain_builder_text(elements_data)
        import_json = _plain_builder_text(import_json)

        dimensions = resolve_dimensions_from_controls(
            size_mode=SIZE_MODE_ASPECT_RATIO,
            max_side=dimension_values.get("max side"),
            aspect_ratio=dimension_values.get("aspect ratio"),
            reference_inputs=None,
            default_width=DEFAULT_MAX_SIDE,
            default_height=DEFAULT_MAX_SIDE,
            multiple_value=dimension_values.get("multiple value"),
        )
        width = int(dimensions.width)
        height = int(dimensions.height)
        caption, boxes, boxes_seeded, used_import = prompt_builder.build_caption(
            background=background,
            style=style,
            high_level_description=high_level_description,
            aesthetics=aesthetics,
            lighting=lighting,
            medium=medium,
            photo=photo,
            art_style=art_style,
            style_palette_data=style_palette_data,
            elements_data=elements_data,
            import_json=import_json,
            import_mode=import_mode,
            coord_mode=coord_mode,
            bbox_order=bbox_order,
            bboxes=bboxes,
            width=width,
            height=height,
        )
        prompt = prompt_builder.format_caption(caption, output_format)
        prompt_output = prompt
        preview = self._render_preview(boxes, width, height, image, int(bg_brightness))
        bboxes_out = prompt_builder.pixel_bboxes(boxes, width, height)
        payload = {
            "family": "ideogram4",
            "prompt": prompt,
            "width": width,
            "height": height,
            "max_side": int(dimensions.max_side),
            "aspect_ratio": dimensions.aspect_ratio,
            "multiple_value": dimensions.multiple_value,
            "output_format": output_format,
            "coord_mode": prompt_builder.sanitize_coord_mode(coord_mode),
            "bbox_order": prompt_builder.sanitize_bbox_order(bbox_order),
            "privacy_mode": bool(privacy_mode),
        }
        ui = {"dims": [width, height]}
        if boxes_seeded and not privacy_mode:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import and not privacy_mode:
            ui["caption"] = [prompt_builder.dumps_pretty(caption)]
        return {
            "ui": ui,
            "result": (payload, prompt_output, preview, bboxes_out, width, height),
        }

    @staticmethod
    def _render_preview(boxes: list[dict[str, Any]], width: int, height: int, image: Any, brightness: int):
        bg = None
        if image is not None:
            try:
                import numpy as np
                from PIL import Image

                bg = Image.fromarray((image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            except Exception:
                bg = None
        return prompt_builder.render_preview(boxes, width, height, bg, brightness)
