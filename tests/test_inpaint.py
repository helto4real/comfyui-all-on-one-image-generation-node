import sys
from types import SimpleNamespace

import pytest

from nodes.inpaint import AIOInpaint
from services.inpaint import (
    apply_inpaint_model_conditioning,
    encode_inpaint_source_latent,
    grow_inpaint_mask,
    InpaintSource,
    normalize_inpaint_config,
    prepare_inpaint_latent,
    prepare_inpaint_mask,
    prepare_inpaint_source,
    resolve_dimensions_from_inpaint_config,
    stitch_inpaint_image,
)


class FakeImage:
    shape = (1, 769, 513, 3)


class FakeMask:
    shape = (1, 769, 513)


class FakeCroppedImage:
    shape = (1, 640, 832, 3)


def test_inpaint_node_exposes_config_schema():
    schema = AIOInpaint.INPUT_TYPES()
    inputs = schema["required"]
    optional = schema["optional"]

    assert inputs["image"][0] == "IMAGE"
    assert inputs["mask"][0] == "MASK"
    assert inputs["mask_invert"][1]["default"] is False
    assert inputs["mask_grow"][1]["default"] == 16
    assert inputs["mask_feather"][1]["default"] == 24
    assert inputs["denoise"][1]["default"] == 1.0
    assert inputs["final_blend"][1]["default"] is True
    assert inputs["crop_target_width"][1]["default"] == 1024
    assert inputs["crop_target_width"][1]["advanced"] is True
    assert inputs["crop_target_height"][1]["default"] == 1024
    assert inputs["context_from_mask_extend_factor"][1]["default"] == 1.6
    assert inputs["crop_output_padding"][1]["default"] == "64"
    assert inputs["mask_fill_holes"][1]["default"] is True
    assert inputs["mask_hipass_filter"][1]["default"] == 0.1
    assert optional["context_mask"][0] == "MASK"


def test_inpaint_config_normalizes_defaults():
    image = FakeImage()
    mask = FakeMask()

    config = normalize_inpaint_config(image=image, mask=mask)

    assert config["version"] == 1
    assert config["image"] is image
    assert config["mask"] is mask
    assert config["mask_invert"] is False
    assert config["mask_grow"] == 16
    assert config["mask_feather"] == 24
    assert config["denoise"] == 1.0
    assert config["final_blend"] is True
    assert config["crop_target_width"] == 1024
    assert config["crop_target_height"] == 1024
    assert config["context_from_mask_extend_factor"] == 1.6
    assert config["crop_output_padding"] == "64"
    assert config["mask_fill_holes"] is True
    assert config["mask_hipass_filter"] == 0.1
    assert config["context_mask"] is None


def test_inpaint_config_preserves_controls():
    context_mask = FakeMask()
    config = AIOInpaint().configure(
        image=FakeImage(),
        mask=FakeMask(),
        mask_invert=True,
        mask_grow=6,
        mask_feather=16,
        denoise=0.45,
        final_blend=False,
        crop_target_width=1280,
        crop_target_height=768,
        context_from_mask_extend_factor=2.0,
        crop_output_padding="128",
        mask_fill_holes=False,
        mask_hipass_filter=0.25,
        context_mask=context_mask,
    )[0]

    assert config["mask_invert"] is True
    assert config["mask_grow"] == 6
    assert config["mask_feather"] == 16
    assert config["denoise"] == 0.45
    assert config["final_blend"] is False
    assert config["crop_target_width"] == 1280
    assert config["crop_target_height"] == 768
    assert config["context_from_mask_extend_factor"] == 2.0
    assert config["crop_output_padding"] == "128"
    assert config["mask_fill_holes"] is False
    assert config["mask_hipass_filter"] == 0.25
    assert config["context_mask"] is context_mask


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
    with pytest.raises(ValueError, match="crop_target_width must be between 64 and 16384"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), crop_target_width=63)
    with pytest.raises(ValueError, match="crop_target_height must be between 64 and 16384"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), crop_target_height=16385)
    with pytest.raises(ValueError, match="mask_hipass_filter must be between 0.0 and 1.0"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), mask_hipass_filter=1.1)


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


def test_prepare_inpaint_source_uses_crop_node_when_available(monkeypatch):
    captured = {}
    cropped_image = FakeCroppedImage()

    class FakeCropNode:
        def inpaint_crop(self, **kwargs):
            captured.update(kwargs)
            return "stitcher", cropped_image, "cropped_mask"

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": FakeCropNode}),
    )

    source = prepare_inpaint_source(
        config=normalize_inpaint_config(
            image="image",
            mask="mask",
            context_mask="context_mask",
            mask_invert=True,
            crop_target_width=1280,
            crop_target_height=768,
            context_from_mask_extend_factor=2.25,
            crop_output_padding="128",
            mask_fill_holes=False,
            mask_hipass_filter=0.25,
        ),
        width=1024,
        height=1024,
    )

    assert source == InpaintSource(
        image=cropped_image,
        mask="cropped_mask",
        noise_mask="cropped_mask",
        stitcher="stitcher",
        used_crop=True,
        width=832,
        height=640,
    )
    assert captured["image"] == "image"
    assert captured["mask"] == "mask"
    assert captured["optional_context_mask"] == "context_mask"
    assert captured["mask_invert"] is True
    assert captured["mask_expand_pixels"] == 16
    assert captured["mask_blend_pixels"] == 24
    assert captured["mask_fill_holes"] is False
    assert captured["mask_hipass_filter"] == 0.25
    assert captured["context_from_mask_extend_factor"] == 2.25
    assert captured["output_resize_to_target_size"] is True
    assert captured["output_target_width"] == 1280
    assert captured["output_target_height"] == 768
    assert captured["output_padding"] == "128"
    assert captured["device_mode"] == "gpu (much faster)"


def test_prepare_inpaint_source_falls_back_to_full_frame_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, 1:3, 1:3] = 1.0

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    source = prepare_inpaint_source(
        config=normalize_inpaint_config(image=image, mask=mask, mask_grow=3),
        width=4,
        height=4,
    )

    assert source.stitcher is None
    assert source.used_crop is False
    assert source.image is image
    assert source.working_dimensions(fallback_width=1, fallback_height=1) == (4, 4)
    assert source.mask.shape == (1, 4, 4)
    assert int(source.mask.sum().item()) == 4
    assert source.noise_mask.shape == (1, 4, 4)
    assert int(source.noise_mask.sum().item()) == 16


def test_encode_inpaint_source_latent_encodes_clean_source_and_attaches_noise_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    blend_mask = torch.zeros((1, 4, 4))
    noise_mask = torch.ones((1, 4, 4))
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

    latent = encode_inpaint_source_latent(
        vae="vae",
        source=InpaintSource(image=image, mask=blend_mask, noise_mask=noise_mask),
    )

    assert captured["vae"] == "vae"
    assert captured["pixels"] is image
    assert latent["samples"] == "clean_source_latent"
    assert latent["noise_mask"].shape == (1, 1, 4, 4)
    assert torch.equal(latent["noise_mask"][:, 0], noise_mask)


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
