import sys
from types import SimpleNamespace

import pytest

from nodes.inpaint import AIOInpaint
from services.inpaint import (
    apply_inpaint_model_conditioning,
    FluxInpaintSource,
    grow_inpaint_mask,
    normalize_inpaint_config,
    prepare_flux_inpaint_source,
    prepare_inpaint_latent,
    prepare_inpaint_mask,
    resolve_dimensions_from_inpaint_config,
    stitch_inpaint_image,
)


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


def test_prepare_inpaint_mask_can_invert_mask():
    torch = pytest.importorskip("torch")
    mask = torch.tensor([[[0.0, 1.0], [0.25, 0.75]]])

    prepared = prepare_inpaint_mask(
        normalize_inpaint_config(image=FakeImage(), mask=mask, mask_invert=True),
        width=2,
        height=2,
    )

    assert torch.allclose(prepared, torch.tensor([[[1.0, 0.0], [0.75, 0.25]]]))


def test_grow_inpaint_mask_expands_rounded_active_area():
    torch = pytest.importorskip("torch")
    mask = torch.zeros((1, 5, 5))
    mask[:, 2, 2] = 1.0

    grown = grow_inpaint_mask(mask, 3)

    assert grown.shape == (1, 5, 5)
    assert int(grown.sum().item()) == 9
    assert grown[0, 1:4, 1:4].min().item() == 1.0


def test_prepare_inpaint_latent_encodes_clean_image_and_attaches_noise_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, 1:3, 1:3] = 1.0
    captured = {}

    class FakeVAEEncode:
        def encode(self, vae, pixels):
            captured["vae"] = vae
            captured["pixels"] = pixels
            return ({"samples": "clean_source_latent"},)

    class FakeVAEEncodeForInpaint:
        def encode(self, *args, **kwargs):
            raise AssertionError("gray-masked VAEEncodeForInpaint should not be used")

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(VAEEncode=FakeVAEEncode, VAEEncodeForInpaint=FakeVAEEncodeForInpaint),
    )

    latent, source_image, blend_mask = prepare_inpaint_latent(
        vae="vae",
        config=normalize_inpaint_config(image=image, mask=mask, mask_grow=0),
        width=4,
        height=4,
    )

    assert captured["vae"] == "vae"
    assert captured["pixels"] is image
    assert source_image is image
    assert torch.equal(blend_mask, mask)
    assert latent["samples"] == "clean_source_latent"
    assert latent["noise_mask"].shape == (1, 1, 4, 4)
    assert torch.equal(latent["noise_mask"][:, 0], mask)


def test_prepare_flux_inpaint_source_uses_crop_node_when_available(monkeypatch):
    captured = {}

    class FakeCropNode:
        def inpaint_crop(self, **kwargs):
            captured.update(kwargs)
            return "stitcher", "cropped_image", "cropped_mask"

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": FakeCropNode}),
    )

    source = prepare_flux_inpaint_source(
        config=normalize_inpaint_config(
            image="image",
            mask="mask",
            mask_invert=True,
            mask_grow=16,
            mask_feather=24,
        ),
        width=1024,
        height=1024,
    )

    assert source == FluxInpaintSource(
        image="cropped_image",
        mask="cropped_mask",
        stitcher="stitcher",
        used_crop=True,
    )
    assert captured["image"] == "image"
    assert captured["mask"] == "mask"
    assert captured["mask_invert"] is True
    assert captured["mask_expand_pixels"] == 16
    assert captured["mask_blend_pixels"] == 24
    assert captured["mask_fill_holes"] is True
    assert captured["mask_hipass_filter"] == 0.1
    assert captured["context_from_mask_extend_factor"] == 1.6
    assert captured["output_resize_to_target_size"] is True
    assert captured["output_target_width"] == 1024
    assert captured["output_target_height"] == 1024
    assert captured["output_padding"] == "64"
    assert captured["device_mode"] == "gpu (much faster)"


def test_prepare_flux_inpaint_source_falls_back_to_full_frame_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, 1:3, 1:3] = 1.0

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    source = prepare_flux_inpaint_source(
        config=normalize_inpaint_config(image=image, mask=mask, mask_grow=3),
        width=4,
        height=4,
    )

    assert source.stitcher is None
    assert source.used_crop is False
    assert source.image is image
    assert source.mask.shape == (1, 4, 4)
    assert int(source.mask.sum().item()) == 16


def test_apply_inpaint_model_conditioning_uses_comfy_node(monkeypatch):
    captured = {}

    class FakeInpaintModelConditioning:
        def encode(self, **kwargs):
            captured.update(kwargs)
            return "positive_out", "negative_out", {"samples": "latent", "noise_mask": "mask"}

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(InpaintModelConditioning=FakeInpaintModelConditioning),
    )

    positive, negative, latent = apply_inpaint_model_conditioning(
        vae="vae",
        positive="positive",
        negative="negative",
        image="image",
        mask="mask",
    )

    assert positive == "positive_out"
    assert negative == "negative_out"
    assert latent == {"samples": "latent", "noise_mask": "mask"}
    assert captured == {
        "positive": "positive",
        "negative": "negative",
        "pixels": "image",
        "vae": "vae",
        "mask": "mask",
        "noise_mask": True,
    }


def test_stitch_inpaint_image_uses_crop_stitch_node(monkeypatch):
    captured = {}

    class FakeStitchNode:
        def inpaint_stitch(self, **kwargs):
            captured.update(kwargs)
            return ("original_size_image",)

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintStitchImproved": FakeStitchNode}),
    )

    image = stitch_inpaint_image(stitcher="stitcher", inpainted_image="decoded_crop")

    assert image == "original_size_image"
    assert captured == {"stitcher": "stitcher", "inpainted_image": "decoded_crop"}
