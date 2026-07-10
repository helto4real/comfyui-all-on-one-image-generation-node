import json
import math
from pathlib import Path

import pytest
from helto_privacy import initialize_keystore

from nodes.ideogram4_prompt_builder import AIOIdeogram4PromptBuilder
from services import privacy
from services import ideogram4_prompt_builder as builder

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "correct horse battery"


def test_compact_json_matches_kj_key_order_and_formatting():
    caption, boxes, boxes_seeded, used_import = builder.build_caption(
        background="Room",
        style="photo",
        high_level_description="Overview",
        aesthetics="realistic",
        lighting="natural",
        medium="photography",
        photo="casual",
        style_palette_data='["#ff0000"]',
        elements_data=json.dumps(
            [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "type": "obj",
                    "desc": "person",
                    "palette": ["#00ff00", "#0000ff"],
                }
            ]
        ),
    )

    assert boxes_seeded is False
    assert used_import is False
    assert boxes[0]["desc"] == "person"
    assert builder.format_caption(caption, "compact") == (
        '{"high_level_description":"Overview",'
        '"style_description":{"aesthetics":"realistic","lighting":"natural","photo":"casual",'
        '"medium":"photography","color_palette":["#FF0000"]},'
        '"compositional_deconstruction":{"background":"Room","elements":[{"type":"obj",'
        '"bbox":[200,100,600,400],"desc":"person","color_palette":["#00FF00","#0000FF"]}]}}'
    )


def test_xy_bbox_order_uses_x_first_grid_coordinates():
    caption, _, _, _ = builder.build_caption(
        background="Room",
        style="none",
        bbox_order="xy",
        elements_data=json.dumps(
            [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "type": "obj",
                    "desc": "person",
                }
            ]
        ),
    )

    assert caption["compositional_deconstruction"]["elements"][0]["bbox"] == [100, 200, 400, 600]


def test_default_editor_color_is_display_only_for_elements():
    caption, _, _, _ = builder.build_caption(
        background="Room",
        style="none",
        elements_data=json.dumps(
            [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "type": "obj",
                    "desc": "person",
                    "palette": ["#8ca8ff"],
                }
            ]
        ),
    )

    assert caption["compositional_deconstruction"]["elements"][0] == {
        "type": "obj",
        "bbox": [200, 100, 600, 400],
        "desc": "person",
    }


def test_editor_payload_supplies_high_level_description_when_widget_value_lags():
    caption, _, _, _ = builder.build_caption(
        background="",
        style="photo",
        high_level_description="",
        elements_data=json.dumps(
            {
                "version": 1,
                "widgets": {
                    "high_level_description": "A beach shot of a man and a woman",
                },
                "elements": [
                    {
                        "x": 0.067,
                        "y": 0.192,
                        "w": 0.307,
                        "h": 0.674,
                        "type": "obj",
                        "desc": "A man",
                    }
                ],
            }
        ),
    )

    assert caption["high_level_description"] == "A beach shot of a man and a woman"
    assert caption["compositional_deconstruction"]["elements"][0]["bbox"] == [192, 67, 866, 374]


def test_absolute_bbox_mode_scales_by_resolved_dimensions():
    caption, _, _, _ = builder.build_caption(
        background="Room",
        style="none",
        coord_mode="absolute",
        bbox_order="xy",
        width=2000,
        height=1000,
        elements_data=json.dumps(
            [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "type": "obj",
                    "desc": "person",
                }
            ]
        ),
    )

    assert caption["compositional_deconstruction"]["elements"][0]["bbox"] == [200, 200, 800, 600]


def test_invalid_bbox_coordinate_options_fall_back_to_ideogram_defaults():
    caption, _, _, _ = builder.build_caption(
        background="Room",
        style="none",
        coord_mode="pixels",
        bbox_order="bad",
        width=2000,
        height=1000,
        elements_data=json.dumps(
            [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "type": "obj",
                    "desc": "person",
                }
            ]
        ),
    )

    assert caption["compositional_deconstruction"]["elements"][0]["bbox"] == [200, 100, 600, 400]


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_private_prompt_builder_encrypts_payload_but_returns_plain_prompt_output(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy, "config_dir", lambda: tmp_path)
    initialize_keystore(PASSWORD)

    result = AIOIdeogram4PromptBuilder().build_prompt(
        high_level_description="Private overview",
        background="Private room",
        style="photo",
        import_mode="when empty",
        output_format="compact",
        bg_brightness=25,
        privacy_mode=True,
        **{"max side": 1024, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]

    payload, prompt_output = result[0], result[1]

    assert "Private overview" not in json.dumps(payload)
    assert "Private room" not in json.dumps(payload)
    assert privacy.is_encrypted_payload(payload["prompt"])
    assert privacy.decrypt_text_if_encrypted(payload["prompt"]) == prompt_output
    assert "Private overview" in prompt_output
    assert "Private room" in prompt_output


def test_prompt_builder_is_changed_uses_native_cache_for_public_inputs(monkeypatch):
    monkeypatch.setattr(privacy, "external_cache_providers_registered", lambda: True)

    assert AIOIdeogram4PromptBuilder.IS_CHANGED(privacy_mode=False) is False


def test_prompt_builder_is_changed_disables_external_cache_for_private_outputs(monkeypatch):
    monkeypatch.setattr(privacy, "external_cache_providers_registered", lambda: True)

    assert math.isnan(AIOIdeogram4PromptBuilder.IS_CHANGED(privacy_mode=True))


def test_prompt_builder_private_ui_omits_caption_and_boxes(monkeypatch):
    monkeypatch.setattr(AIOIdeogram4PromptBuilder, "_render_preview", staticmethod(lambda *args: "preview"))
    monkeypatch.setattr(privacy, "encrypt_state", lambda state: {"encrypted": True, "state": state})

    output = AIOIdeogram4PromptBuilder().build_prompt(
        import_json=json.dumps(
            {
                "compositional_deconstruction": {
                    "background": "Private room",
                    "elements": [{"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": "Private person"}],
                }
            }
        ),
        import_mode="always",
        privacy_mode=True,
        **{"max side": 1024, "aspect ratio": "1:1", "multiple value": "none"},
    )
    boxes_output = AIOIdeogram4PromptBuilder().build_prompt(
        bboxes=[{"x": 100, "y": 100, "width": 300, "height": 300}],
        privacy_mode=True,
        **{"max side": 1000, "aspect ratio": "1:1", "multiple value": "none"},
    )

    assert output["ui"] == {"dims": [1024, 1024]}
    assert boxes_output["ui"] == {"dims": [1000, 1000]}


def test_pretty_json_matches_kj_scalar_array_formatting():
    caption, _, _, _ = builder.build_caption(
        background="Room",
        style="none",
        elements_data=json.dumps(
            [
                {
                    "x": 0.0,
                    "y": 0.0,
                    "w": 1.0,
                    "h": 1.0,
                    "type": "text",
                    "text": "SALE",
                    "desc": "large letters",
                    "palette": ["#ffffff"],
                }
            ]
        ),
    )

    assert builder.format_caption(caption, "pretty") == (
        '{\n'
        '    "compositional_deconstruction": {\n'
        '        "background": "Room",\n'
        '        "elements": [\n'
        '            {\n'
        '                "type": "text",\n'
        '                "bbox": [0, 0, 1000, 1000],\n'
        '                "text": "SALE",\n'
        '                "desc": "large letters",\n'
        '                "color_palette": ["#FFFFFF"]\n'
        '            }\n'
        '        ]\n'
        '    }\n'
        '}'
    )


def test_art_style_and_palette_truncation_match_kj_shape():
    caption, _, _, _ = builder.build_caption(
        background="Backdrop",
        style={"style": "art_style", "art_style": "ink"},
        aesthetics="clean",
        lighting="flat",
        medium="poster",
        elements_data=json.dumps(
            [
                {
                    "nobbox": True,
                    "type": "obj",
                    "desc": "symbol",
                    "palette": ["#111111", "#222222", "#333333", "#444444", "#555555", "#666666"],
                }
            ]
        ),
    )

    assert caption["style_description"] == {
        "aesthetics": "clean",
        "lighting": "flat",
        "medium": "poster",
        "art_style": "ink",
    }
    assert caption["compositional_deconstruction"]["elements"][0] == {
        "type": "obj",
        "desc": "symbol",
        "color_palette": ["#111111", "#222222", "#333333", "#444444", "#555555"],
    }


def test_import_json_can_drive_output_and_boxes():
    source = {
        "compositional_deconstruction": {
            "background": "Imported",
            "elements": [{"type": "obj", "bbox": [100, 200, 300, 500], "desc": "item"}],
        }
    }

    caption, boxes, _, used_import = builder.build_caption(
        background="Ignored",
        style="none",
        import_json=json.dumps(source),
        import_mode="always",
    )

    assert used_import is True
    assert caption == source
    assert boxes == [{"type": "obj", "text": "", "desc": "item", "palette": [], "x": 0.2, "y": 0.1, "w": 0.3, "h": 0.2}]


def test_import_json_parses_xy_bbox_order_for_editor_boxes():
    source = {
        "compositional_deconstruction": {
            "background": "Imported",
            "elements": [{"type": "obj", "bbox": [100, 200, 400, 600], "desc": "item"}],
        }
    }

    caption, boxes, _, used_import = builder.build_caption(
        background="Ignored",
        style="none",
        import_json=json.dumps(source),
        import_mode="always",
        bbox_order="xy",
    )

    assert used_import is True
    assert caption == source
    assert boxes == [{"type": "obj", "text": "", "desc": "item", "palette": [], "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]


def test_builder_node_serialized_elements_emit_all_json_elements(monkeypatch):
    monkeypatch.setattr(
        AIOIdeogram4PromptBuilder,
        "_render_preview",
        staticmethod(lambda boxes, width, height, image, brightness: "preview"),
    )
    elements = json.dumps(
        [
            {
                "x": 0.0,
                "y": 0.0,
                "w": 0.25,
                "h": 0.5,
                "type": "obj",
                "desc": "left panel",
                "palette": ["#111111"],
            },
            {
                "x": 0.35,
                "y": 0.25,
                "w": 0.5,
                "h": 0.4,
                "type": "text",
                "text": "HELTO",
                "desc": "right label",
                "palette": ["#222222"],
            },
        ]
    )

    _, prompt, _, bboxes, _, _ = AIOIdeogram4PromptBuilder().build_prompt(
        high_level_description="Two panel layout",
        background="studio wall",
        style="none",
        elements_data=elements,
        **{"max side": 1000, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]

    parsed = json.loads(prompt)
    assert parsed["high_level_description"] == "Two panel layout"
    assert parsed["compositional_deconstruction"]["background"] == "studio wall"
    assert parsed["compositional_deconstruction"]["elements"] == [
        {"type": "obj", "bbox": [0, 0, 500, 250], "desc": "left panel", "color_palette": ["#111111"]},
        {
            "type": "text",
            "bbox": [250, 350, 650, 850],
            "text": "HELTO",
            "desc": "right label",
            "color_palette": ["#222222"],
        },
    ]
    assert bboxes == [[{"x": 0, "y": 0, "width": 250, "height": 500}, {"x": 350, "y": 250, "width": 500, "height": 400}]]


def test_bbox_seed_and_pixel_output_match_kj_shape():
    _, boxes, boxes_seeded, _ = builder.build_caption(
        background="Room",
        style="none",
        bboxes=[{"x": 100, "y": 50, "width": 300, "height": 200}],
        width=1000,
        height=500,
    )

    assert boxes_seeded is True
    assert boxes[0]["x"] == 0.1
    assert boxes[0]["y"] == 0.1
    assert builder.pixel_bboxes(boxes, 1000, 500) == [[{"x": 100, "y": 50, "width": 300, "height": 200}]]


def test_prompt_builder_node_resolves_generate_style_dimensions(monkeypatch):
    monkeypatch.setattr(
        AIOIdeogram4PromptBuilder,
        "_render_preview",
        staticmethod(lambda boxes, width, height, image, brightness: "preview"),
    )

    payload, prompt, preview, bboxes, width, height = AIOIdeogram4PromptBuilder().build_prompt(
        background="Room",
        style="none",
        **{"max side": 1088, "aspect ratio": "16:9", "multiple value": "16"},
    )["result"]

    assert (width, height) == (1088, 608)
    assert payload["width"] == 1088
    assert payload["height"] == 608
    assert payload["prompt"] == prompt
    assert preview == "preview"
    assert bboxes == []


def test_prompt_builder_node_exposes_privacy_mode():
    inputs = AIOIdeogram4PromptBuilder.INPUT_TYPES()

    assert inputs["required"]["privacy_mode"][1]["default"] is False


def test_prompt_builder_node_exposes_coordinate_options():
    inputs = AIOIdeogram4PromptBuilder.INPUT_TYPES()

    assert inputs["required"]["coord_mode"][0] == ["normalized", "absolute"]
    assert inputs["required"]["coord_mode"][1]["default"] == "normalized"
    assert inputs["required"]["bbox_order"][0] == ["yx", "xy"]
    assert inputs["required"]["bbox_order"][1]["default"] == "yx"


def test_prompt_builder_node_applies_xy_coordinate_options(monkeypatch):
    monkeypatch.setattr(
        AIOIdeogram4PromptBuilder,
        "_render_preview",
        staticmethod(lambda boxes, width, height, image, brightness: "preview"),
    )
    elements = json.dumps(
        [
            {
                "x": 0.1,
                "y": 0.2,
                "w": 0.3,
                "h": 0.4,
                "type": "obj",
                "desc": "person",
            }
        ]
    )

    payload, prompt, *_ = AIOIdeogram4PromptBuilder().build_prompt(
        background="Room",
        style="none",
        coord_mode="normalized",
        bbox_order="xy",
        elements_data=elements,
        **{"max side": 1000, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]

    assert json.loads(prompt)["compositional_deconstruction"]["elements"][0]["bbox"] == [100, 200, 400, 600]
    assert payload["coord_mode"] == "normalized"
    assert payload["bbox_order"] == "xy"


def test_prompt_builder_frontend_syncs_native_text_and_display_only_palette():
    source = (ROOT / "web/js/aio_ideogram4_prompt_builder.js").read_text(encoding="utf-8")

    assert "function parseElementsPayload(value)" in source
    assert "function captionWidgetValues()" in source
    assert "function widgetDomTextValue(widget)" in source
    assert "function syncLiveWidgetTextValue(widget)" in source
    assert "syncLiveWidgetTextValues();" in source
    assert "syncLiveWidgetTextValue(widget);" in source
    assert "setExecutionWidgetValue(elementsWidget, serializedElementsValue());" in source
    assert "return serializePrivateValue(this, liveWidgetValue(this));" in source
    assert "function promptPalette(colors)" in source
    assert "values.length === 1 && values[0] === DEFAULT_COLOR_UPPER ? [] : values" in source
    assert "palette: []," in source
    assert 'boxes[active].palette.push("#FFFFFF");' in source


def test_prompt_builder_decrypts_private_fields_without_changing_output(monkeypatch, tmp_path):
    from nodes import ideogram4_prompt_builder as node_module
    from services import privacy

    monkeypatch.setattr(node_module.privacy, "config_dir", lambda: tmp_path)
    initialize_keystore(PASSWORD)
    monkeypatch.setattr(
        AIOIdeogram4PromptBuilder,
        "_render_preview",
        staticmethod(lambda boxes, width, height, image, brightness: "preview"),
    )
    elements = json.dumps(
        [
            {
                "x": 0.1,
                "y": 0.2,
                "w": 0.3,
                "h": 0.4,
                "type": "obj",
                "desc": "person",
            },
            {
                "x": 0.5,
                "y": 0.1,
                "w": 0.2,
                "h": 0.2,
                "type": "text",
                "text": "PRIVATE",
                "desc": "sign",
            }
        ]
    )
    encrypted_background = json.dumps(privacy.encrypt_state({"value": "Private room"}))
    encrypted_elements = json.dumps(privacy.encrypt_state({"value": elements}))

    payload, prompt, *_ = AIOIdeogram4PromptBuilder().build_prompt(
        background=encrypted_background,
        style="none",
        elements_data=encrypted_elements,
        privacy_mode=True,
        **{"max side": 1024, "aspect ratio": "1:1", "multiple value": "16"},
    )["result"]

    expected = (
        '{"compositional_deconstruction":{"background":"Private room",'
        '"elements":[{"type":"obj","bbox":[200,100,600,400],"desc":"person"},'
        '{"type":"text","bbox":[100,500,300,700],"text":"PRIVATE","desc":"sign"}]}}'
    )
    assert privacy.decrypt_text_if_encrypted(payload["prompt"]) == expected
    assert prompt == expected
