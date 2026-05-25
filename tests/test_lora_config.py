import pytest

from nodes.lora_configuration import AIOLoraConfiguration
from services.lora_config import normalize_lora_config


def test_lora_config_node_returns_custom_type():
    assert AIOLoraConfiguration.RETURN_TYPES == ("AIO_LORA_CONFIG",)
    assert AIOLoraConfiguration.CATEGORY == "AIO/Image"


def test_lora_config_node_normalizes_rgthree_style_rows():
    config = AIOLoraConfiguration().configure(
        show_strengths="separate",
        match="style",
        lora_2={
            "on": True,
            "lora": "detail",
            "strength": 0.4,
            "strengthTwo": 0.2,
        },
        lora_1={
            "on": True,
            "lora": "style",
            "strength": 1.25,
        },
    )[0]

    assert config["version"] == 1
    assert config["ui"] == {"show_strengths": "separate", "match": "style"}
    assert config["loras"] == [
        {
            "enabled": True,
            "name": "style",
            "strength_model": 1.25,
            "strength_clip": 1.25,
        },
        {
            "enabled": True,
            "name": "detail",
            "strength_model": 0.4,
            "strength_clip": 0.2,
        },
    ]


def test_lora_config_skips_disabled_and_zero_strength_rows():
    config = normalize_lora_config(
        {
            "lora_1": {"on": False, "lora": "off", "strength": 1.0},
            "lora_2": {"on": True, "lora": "zero", "strength": 0, "strengthTwo": 0},
            "lora_3": {"on": True, "lora": "keep", "strength": 0, "strengthTwo": 0.6},
        }
    )

    assert [lora["name"] for lora in config["loras"]] == ["keep"]


def test_lora_config_rejects_unknown_lora_when_lora_list_available():
    with pytest.raises(ValueError, match="was selected, but it was not found"):
        normalize_lora_config(
            {"lora_1": {"on": True, "lora": "missing", "strength": 1}},
            available_loras=["known.safetensors"],
        )


def test_lora_config_matches_basename_without_extension():
    config = normalize_lora_config(
        {"lora_1": {"on": True, "lora": "style", "strength": 1}},
        available_loras=["subdir/style.safetensors"],
    )

    assert config["loras"][0]["name"] == "subdir/style.safetensors"
