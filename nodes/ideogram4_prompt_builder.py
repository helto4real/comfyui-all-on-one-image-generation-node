"""Classic ComfyUI node for building Ideogram 4 structured prompts."""

from __future__ import annotations

import json
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
    from ..services import privacy
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
    from services import privacy


STYLE_OPTIONS = ("none", "photo", "art_style")
IMPORT_MODES = ("when empty", "always")
OUTPUT_FORMATS = ("compact", "pretty")


class AIOIdeogram4PromptBuilder:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_IDEOGRAM4_PROMPT", "STRING", "IMAGE", "BOUNDING_BOX", "INT", "INT")
    RETURN_NAMES = ("prompt_builder", "prompt", "preview", "bboxes", "width", "height")
    FUNCTION = "build_prompt"

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
            },
            "optional": {
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
        style_palette_data: str = "",
        elements_data: str = "",
        bg_brightness: int = 25,
        image: Any = None,
        import_json: str = "",
        bboxes: Any = None,
        privacy_mode: bool = False,
        **dimension_values: Any,
    ):
        del privacy_mode
        high_level_description = privacy.decrypt_text_if_encrypted(high_level_description)
        background = privacy.decrypt_text_if_encrypted(background)
        photo = privacy.decrypt_text_if_encrypted(photo)
        art_style = privacy.decrypt_text_if_encrypted(art_style)
        aesthetics = privacy.decrypt_text_if_encrypted(aesthetics)
        lighting = privacy.decrypt_text_if_encrypted(lighting)
        medium = privacy.decrypt_text_if_encrypted(medium)
        style_palette_data = privacy.decrypt_text_if_encrypted(style_palette_data)
        elements_data = privacy.decrypt_text_if_encrypted(elements_data)
        import_json = privacy.decrypt_text_if_encrypted(import_json)

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
            bboxes=bboxes,
            width=width,
            height=height,
        )
        prompt = prompt_builder.format_caption(caption, output_format)
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
        }
        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:
            ui["caption"] = [prompt_builder.dumps_pretty(caption)]
        return {
            "ui": ui,
            "result": (payload, prompt, preview, bboxes_out, width, height),
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
