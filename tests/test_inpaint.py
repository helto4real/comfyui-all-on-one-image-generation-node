import pytest

from nodes.inpaint import AIOInpaint
from services.inpaint import normalize_inpaint_config, resolve_dimensions_from_inpaint_config


class FakeImage:
    shape = (1, 769, 513, 3)


class FakeMask:
    shape = (1, 769, 513)


def test_inpaint_node_exposes_config_schema():
    inputs = AIOInpaint.INPUT_TYPES()["required"]

    assert inputs["image"][0] == "IMAGE"
    assert inputs["mask"][0] == "MASK"
    assert inputs["mask_invert"][1]["default"] is False
    assert inputs["mask_grow"][1]["default"] == 6
    assert inputs["mask_feather"][1]["default"] == 16
    assert inputs["denoise"][1]["default"] == 1.0
    assert inputs["final_blend"][1]["default"] is True


def test_inpaint_config_normalizes_defaults():
    image = FakeImage()
    mask = FakeMask()

    config = normalize_inpaint_config(image=image, mask=mask)

    assert config["version"] == 1
    assert config["image"] is image
    assert config["mask"] is mask
    assert config["mask_invert"] is False
    assert config["mask_grow"] == 6
    assert config["mask_feather"] == 16
    assert config["denoise"] == 1.0
    assert config["final_blend"] is True


def test_inpaint_config_preserves_controls():
    config = AIOInpaint().configure(
        image=FakeImage(),
        mask=FakeMask(),
        mask_invert=True,
        mask_grow=12,
        mask_feather=32,
        denoise=0.45,
        final_blend=False,
    )[0]

    assert config["mask_invert"] is True
    assert config["mask_grow"] == 12
    assert config["mask_feather"] == 32
    assert config["denoise"] == 0.45
    assert config["final_blend"] is False


def test_inpaint_config_requires_image_and_mask():
    with pytest.raises(ValueError, match="inpaint image is required"):
        normalize_inpaint_config(mask=FakeMask())
    with pytest.raises(ValueError, match="inpaint mask is required"):
        normalize_inpaint_config(image=FakeImage())


def test_inpaint_config_rejects_out_of_range_controls():
    with pytest.raises(ValueError, match="mask_grow must be between 0 and 64"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), mask_grow=65)
    with pytest.raises(ValueError, match="mask_feather must be between 0 and 256"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), mask_feather=257)
    with pytest.raises(ValueError, match="denoise must be between 0.0 and 1.0"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), denoise=1.1)


def test_inpaint_dimensions_round_to_model_multiple():
    dimensions = resolve_dimensions_from_inpaint_config(
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask()),
        multiple=16,
    )

    assert (dimensions.width, dimensions.height) == (512, 768)
    assert dimensions.size_mode == "use inpaint image size"
    assert dimensions.multiple_value == "16"
