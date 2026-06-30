"""Ideogram 4 structured prompt helpers.

The caption-shaping logic in this module is adapted from KJNodes'
Ideogram4PromptBuilderKJ node so AIO can produce byte-for-byte compatible JSON
for equivalent inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FONT_PATH = Path(__file__).resolve().parents[1] / "fonts" / "FreeMono.ttf"
COORD_MODE_NORMALIZED = "normalized"
COORD_MODE_ABSOLUTE = "absolute"
COORD_MODES = (COORD_MODE_NORMALIZED, COORD_MODE_ABSOLUTE)
BBOX_ORDER_YX = "yx"
BBOX_ORDER_XY = "xy"
BBOX_ORDERS = (BBOX_ORDER_YX, BBOX_ORDER_XY)


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    ) if len(value) == 6 else (255, 255, 255)


def readable(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 130:
        t = (130 - lum) / (255 - lum)
        r = round(r + (255 - r) * t)
        g = round(g + (255 - g) * t)
        b = round(b + (255 - b) * t)
    return (r, g, b)


def font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def wrap_text(draw: Any, text: str, font_obj: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font_obj) > max_width:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def render_preview(
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
    bg: Any = None,
    brightness: int = 50,
):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageEnhance

    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        if brightness < 100:
            base = ImageEnhance.Brightness(base).enhance(max(0.0, brightness / 100.0))
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))

    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font_obj = font(fs)
    tag_font = font(max(9, fs - 2))
    line_height = fs + 2

    for index, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        palette_values = [c for c in (box.get("palette") or []) if c]
        r, g, b = hex_rgb(palette_values[0]) if palette_values else (140, 140, 140)
        x1 = max(0, min(rw, round(box.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(box.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((box.get("x", 0) + box.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((box.get("y", 0) + box.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        pal5 = palette_values[:5]
        if pal5 and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(pal5)
            for p, hexc in enumerate(pal5):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=hex_rgb(hexc))

        tag = str(index + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)

        body = box.get("desc", "") or ""
        if box.get("type") == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in wrap_text(draw, body, font_obj, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=readable((r, g, b)) + (255,), font=font_obj)
                ty += line_height

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def sanitize_coord_mode(value: Any) -> str:
    return COORD_MODE_ABSOLUTE if value == COORD_MODE_ABSOLUTE else COORD_MODE_NORMALIZED


def sanitize_bbox_order(value: Any) -> str:
    return BBOX_ORDER_XY if value == BBOX_ORDER_XY else BBOX_ORDER_YX


def bbox_scales(coord_mode: Any, width: int, height: int) -> tuple[int, int]:
    if sanitize_coord_mode(coord_mode) == COORD_MODE_ABSOLUTE:
        return max(1, int(width or 1)), max(1, int(height or 1))
    return 1000, 1000


def norm_bbox(box: dict[str, Any], sx: int = 1000, sy: int = 1000, order: str = BBOX_ORDER_YX) -> list[int]:
    sx = max(1, int(sx or 1))
    sy = max(1, int(sy or 1))
    order = sanitize_bbox_order(order)

    def clamp_x(value: float) -> int:
        return max(0, min(sx, round(value * sx)))

    def clamp_y(value: float) -> int:
        return max(0, min(sy, round(value * sy)))

    x = box.get("x", 0.0)
    y = box.get("y", 0.0)
    w = box.get("w", 0.0)
    h = box.get("h", 0.0)
    ymin, xmin, ymax, xmax = clamp_y(y), clamp_x(x), clamp_y(y + h), clamp_x(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if order == BBOX_ORDER_XY:
        return [xmin, ymin, xmax, ymax]
    return [ymin, xmin, ymax, xmax]


def palette(colors: Any) -> list[str]:
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if c]


def dumps_pretty(value: Any, level: int = 0) -> str:
    pad = "    " * (level + 1)
    end = "    " * level
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(dumps_pretty(item, level) for item in value) + "]"
        return "[\n" + ",\n".join(pad + dumps_pretty(item, level + 1) for item in value) + "\n" + end + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            pad + json.dumps(key, ensure_ascii=False) + ": " + dumps_pretty(val, level + 1)
            for key, val in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(value, ensure_ascii=False)


def dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_list(value: str | None) -> list[Any]:
    if value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def caption_to_boxes(
    caption: dict[str, Any],
    *,
    coord_mode: str = COORD_MODE_NORMALIZED,
    bbox_order: str = BBOX_ORDER_YX,
    width: int = 1024,
    height: int = 1024,
) -> list[dict[str, Any]]:
    cd = caption.get("compositional_deconstruction") or {}
    boxes: list[dict[str, Any]] = []
    sx, sy = bbox_scales(coord_mode, width, height)
    bbox_order = sanitize_bbox_order(bbox_order)
    for element in (cd.get("elements") or []):
        if not isinstance(element, dict):
            continue
        box = {
            "type": "text" if element.get("type") == "text" else "obj",
            "text": element.get("text", "") or "",
            "desc": element.get("desc", "") or "",
            "palette": list(element.get("color_palette") or []),
        }
        bbox = element.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if bbox_order == BBOX_ORDER_XY:
                xmin, ymin, xmax, ymax = bbox
            else:
                ymin, xmin, ymax, xmax = bbox
            box.update(
                x=xmin / sx,
                y=ymin / sy,
                w=(xmax - xmin) / sx,
                h=(ymax - ymin) / sy,
            )
        else:
            box.update(x=0.03, y=0.03, w=0.22, h=0.14, nobbox=True)
        boxes.append(box)
    return boxes


def build_caption(
    *,
    background: str,
    style: str | dict[str, Any],
    high_level_description: str = "",
    aesthetics: str = "",
    lighting: str = "",
    medium: str = "",
    photo: str = "",
    art_style: str = "",
    style_palette_data: str = "",
    elements_data: str = "",
    import_json: str = "",
    import_mode: str = "when empty",
    coord_mode: str = COORD_MODE_NORMALIZED,
    bbox_order: str = BBOX_ORDER_YX,
    bboxes: Any = None,
    width: int = 1024,
    height: int = 1024,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool, bool]:
    if import_mode not in ("when empty", "always"):
        import_mode = "when empty"
    coord_mode = sanitize_coord_mode(coord_mode)
    bbox_order = sanitize_bbox_order(bbox_order)
    bbox_sx, bbox_sy = bbox_scales(coord_mode, width, height)

    boxes = parse_json_list(elements_data)
    boxes_seeded = False
    if not boxes and bboxes:
        if isinstance(bboxes, dict):
            frame = [bboxes]
        elif bboxes and isinstance(bboxes[0], (list, tuple)):
            frame = bboxes[0]
        else:
            frame = bboxes
        for bbox in frame:
            if not isinstance(bbox, dict):
                continue
            boxes.append(
                {
                    "x": bbox.get("x", 0) / width,
                    "y": bbox.get("y", 0) / height,
                    "w": bbox.get("width", 0) / width,
                    "h": bbox.get("height", 0) / height,
                    "type": "obj",
                    "text": "",
                    "desc": "",
                    "palette": [],
                }
            )
        boxes_seeded = bool(boxes)

    imported = None
    if import_json and import_json.strip():
        try:
            candidate = json.loads(import_json)
            if isinstance(candidate, dict):
                imported = candidate
        except json.JSONDecodeError:
            pass

    if isinstance(style, dict):
        kind = str(style.get("style", "none"))
        photo_value = str(style.get("photo", photo))
        art_style_value = str(style.get("art_style", art_style))
    else:
        kind = str(style)
        photo_value = photo
        art_style_value = art_style

    used_import = imported is not None and (import_mode == "always" or not boxes)
    if used_import:
        return (
            imported,
            caption_to_boxes(
                imported,
                coord_mode=coord_mode,
                bbox_order=bbox_order,
                width=width,
                height=height,
            ),
            boxes_seeded,
            True,
        )

    caption: dict[str, Any] = {}
    if high_level_description.strip():
        caption["high_level_description"] = high_level_description

    if kind != "none":
        style_description = {"aesthetics": aesthetics, "lighting": lighting}
        if kind == "photo":
            style_description["photo"] = photo_value
            style_description["medium"] = medium
        else:
            style_description["medium"] = medium
            style_description["art_style"] = art_style_value
        style_palette = palette(parse_json_list(style_palette_data))
        if style_palette:
            style_description["color_palette"] = style_palette
        caption["style_description"] = style_description

    elements: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        element_type = "text" if box.get("type") == "text" else "obj"
        element: dict[str, Any] = {"type": element_type}
        if not box.get("nobbox"):
            element["bbox"] = norm_bbox(box, bbox_sx, bbox_sy, bbox_order)
        if element_type == "text":
            element["text"] = box.get("text", "")
        element["desc"] = box.get("desc", "")
        box_palette = palette(box.get("palette", []))
        if box_palette:
            element["color_palette"] = box_palette[:5]
        elements.append(element)

    caption["compositional_deconstruction"] = {
        "background": background,
        "elements": elements,
    }
    return caption, boxes, boxes_seeded, False


def format_caption(caption: dict[str, Any], output_format: str) -> str:
    return dumps_pretty(caption) if output_format == "pretty" else dumps_compact(caption)


def pixel_bboxes(boxes: list[dict[str, Any]], width: int, height: int) -> list[list[dict[str, int]]]:
    bbox_dicts: list[dict[str, int]] = []
    for box in boxes:
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        x = box.get("x", 0.0)
        y = box.get("y", 0.0)
        box_width = box.get("w", 0.0)
        box_height = box.get("h", 0.0)
        if box_width < 0:
            x += box_width
            box_width = -box_width
        if box_height < 0:
            y += box_height
            box_height = -box_height
        bbox_dicts.append(
            {
                "x": round(x * width),
                "y": round(y * height),
                "width": round(box_width * width),
                "height": round(box_height * height),
            }
        )
    return [bbox_dicts] if bbox_dicts else []
