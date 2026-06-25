import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from nodes.inpaint import AIOInpaint
from services.inpaint import (
    apply_inpaint_color_match,
    apply_inpaint_model_conditioning,
    encode_inpaint_source_latent,
    feather_inpaint_mask,
    grow_inpaint_mask,
    InpaintSource,
    normalize_inpaint_config,
    prepare_inpaint_latent,
    prepare_inpaint_mask,
    prepare_inpaint_output_mask,
    prepare_inpaint_source,
    resolve_dimensions_from_inpaint_config,
    resolve_mask_grow_pixels,
    stitch_inpaint_image,
    stitcher_blend_mask,
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

    assert AIOInpaint.RETURN_TYPES == ("AIO_INPAINT_CONFIG", "MASK")
    assert AIOInpaint.RETURN_NAMES == ("inpaint", "final_mask")
    assert inputs["image"][0] == "IMAGE"
    assert inputs["mask"][0] == "MASK"
    assert inputs["mask_invert"][1]["default"] is False
    assert "mask_grow" not in inputs
    assert inputs["mask_grow_percent"][1]["default"] == 8.0
    assert inputs["mask_grow_percent"][1]["max"] == 100.0
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
    assert inputs["max_full_frame_megapixels"][1]["default"] == 1.0
    assert inputs["max_full_frame_megapixels"][1]["advanced"] is True
    assert inputs["max_full_frame_side"][1]["default"] == 1536
    assert inputs["max_full_frame_side"][1]["advanced"] is True
    assert inputs["color_match_strength"][1]["default"] == 0.0
    assert inputs["color_match_strength"][1]["max"] == 1.0
    assert inputs["color_match_strength"][1]["advanced"] is True
    assert optional["context_mask"][0] == "MASK"


def test_inpaint_config_normalizes_defaults():
    image = FakeImage()
    mask = FakeMask()

    config = normalize_inpaint_config(image=image, mask=mask)

    assert config["version"] == 1
    assert config["image"] is image
    assert config["mask"] is mask
    assert config["mask_invert"] is False
    assert config["mask_grow_percent"] == 8.0
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
    assert config["max_full_frame_megapixels"] == 1.0
    assert config["max_full_frame_side"] == 1536
    assert config["color_match_strength"] == 0.0


def test_inpaint_config_preserves_controls(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    context_mask = FakeMask()

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    config, output_mask = AIOInpaint().configure(
        image=image,
        mask=mask,
        mask_invert=True,
        mask_grow_percent=25.0,
        mask_feather=16,
        denoise=0.45,
        final_blend=False,
        crop_target_width=1280,
        crop_target_height=768,
        context_from_mask_extend_factor=2.0,
        crop_output_padding="128",
        mask_fill_holes=False,
        mask_hipass_filter=0.25,
        max_full_frame_megapixels=2.0,
        max_full_frame_side=2048,
        color_match_strength=0.35,
        context_mask=context_mask,
    )

    assert config["mask_invert"] is True
    assert config["image"] is image
    assert config["mask"] is mask
    assert config["mask_grow_percent"] == 25.0
    assert config["mask_feather"] == 16
    assert config["denoise"] == 0.45
    assert config["final_blend"] is False
    assert config["crop_target_width"] == 1280
    assert config["crop_target_height"] == 768
    assert config["context_from_mask_extend_factor"] == 2.0
    assert config["crop_output_padding"] == "128"
    assert config["mask_fill_holes"] is False
    assert config["mask_hipass_filter"] == 0.25
    assert config["max_full_frame_megapixels"] == 2.0
    assert config["max_full_frame_side"] == 2048
    assert config["color_match_strength"] == 0.35
    assert config["context_mask"] is context_mask
    assert output_mask.shape == (1, 4, 4)


def test_inpaint_config_requires_image_and_mask():
    with pytest.raises(ValueError, match="inpaint image is required"):
        normalize_inpaint_config(mask=FakeMask())
    with pytest.raises(ValueError, match="inpaint mask is required"):
        normalize_inpaint_config(image=FakeImage())


def test_inpaint_config_rejects_out_of_range_controls():
    with pytest.raises(ValueError, match="mask_grow_percent must be between 0.0 and 100.0"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), mask_grow_percent=100.1)
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
    with pytest.raises(ValueError, match="max_full_frame_megapixels must be between 0.25 and 1024.0"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), max_full_frame_megapixels=0.0)
    with pytest.raises(ValueError, match="max_full_frame_side must be between 64 and 16384"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), max_full_frame_side=63)
    with pytest.raises(ValueError, match="color_match_strength must be between 0.0 and 1.0"):
        normalize_inpaint_config(image=FakeImage(), mask=FakeMask(), color_match_strength=1.1)


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


def test_resolve_mask_grow_pixels_uses_active_bbox_percent():
    torch = pytest.importorskip("torch")
    mask = torch.zeros((1, 20, 30))
    mask[:, 4:9, 10:20] = 1.0

    grow_pixels = resolve_mask_grow_pixels(
        normalize_inpaint_config(image=FakeImage(), mask=mask, mask_grow_percent=50.0),
        mask=mask,
        width=30,
        height=20,
    )

    assert grow_pixels == 5


def test_feather_inpaint_mask_softens_active_area():
    torch = pytest.importorskip("torch")
    mask = torch.zeros((1, 5, 5))
    mask[:, 1:4, 1:4] = 1.0

    feathered = feather_inpaint_mask(mask, 2)

    assert feathered.shape == (1, 5, 5)
    assert feathered.max().item() < 1.0
    assert 0.0 < feathered[0, 0, 0].item() < feathered[0, 2, 2].item()


def test_inpaint_node_outputs_processed_mask_with_grow_and_feather(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 20, 20, 3))
    mask = torch.zeros((1, 20, 20))
    mask[:, 8:12, 8:12] = 1.0

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    config, output_mask = AIOInpaint().configure(
        image=image,
        mask=mask,
        mask_grow_percent=75.0,
        mask_feather=2,
    )

    assert config["mask"] is mask
    assert output_mask.shape == (1, 20, 20)
    assert int(mask.sum().item()) == 16
    assert output_mask.sum().item() > 16.0
    assert (output_mask > 0.0).sum().item() > 16
    assert torch.any((output_mask > 0.0) & (output_mask < 1.0))


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
        config=normalize_inpaint_config(image=image, mask=mask, mask_grow_percent=0.0),
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
    image = FakeImage()
    mask = FakeMask()

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
            image=image,
            mask=mask,
            context_mask="context_mask",
            mask_invert=True,
            mask_grow_percent=10.0,
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
    assert captured["image"] is image
    assert captured["mask"] is mask
    assert captured["optional_context_mask"] == "context_mask"
    assert captured["mask_invert"] is True
    assert captured["mask_expand_pixels"] == 77
    assert captured["mask_blend_pixels"] == 24
    assert captured["mask_fill_holes"] is False
    assert captured["mask_hipass_filter"] == 0.25
    assert captured["context_from_mask_extend_factor"] == 2.25
    assert captured["output_resize_to_target_size"] is True
    assert captured["output_target_width"] == 1280
    assert captured["output_target_height"] == 768
    assert captured["output_padding"] == "128"
    assert captured["device_mode"] == "gpu (much faster)"


def test_prepare_inpaint_source_keeps_gpu_crop_for_large_images(monkeypatch):
    captured = {}

    class LargeImage:
        shape = (1, 4096, 4096, 3)

    class FakeCropNode:
        def inpaint_crop(self, **kwargs):
            captured.update(kwargs)
            return "stitcher", FakeCroppedImage(), "cropped_mask"

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": FakeCropNode}),
    )

    prepare_inpaint_source(
        config=normalize_inpaint_config(image=LargeImage(), mask=FakeMask()),
        width=4096,
        height=4096,
    )

    assert captured["device_mode"] == "gpu (much faster)"


def test_prepare_inpaint_source_allows_explicit_cpu_crop(monkeypatch):
    captured = {}

    class FakeCropNode:
        def inpaint_crop(self, **kwargs):
            captured.update(kwargs)
            return "stitcher", FakeCroppedImage(), "cropped_mask"

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": FakeCropNode}),
    )

    prepare_inpaint_source(
        config=normalize_inpaint_config(
            {
                "image": FakeImage(),
                "mask": FakeMask(),
                "crop_device_mode": "cpu (compatible)",
            }
        ),
        width=1024,
        height=1024,
    )

    assert captured["device_mode"] == "cpu (compatible)"


def test_prepare_inpaint_output_mask_uses_stitcher_blend_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}
    blend_mask = torch.full((1, 4, 4), 0.25)

    class SourceImage:
        shape = (1, 6, 10, 3)

    class SourceMask:
        shape = (1, 6, 10)

    class FakeCropNode:
        def inpaint_crop(self, **kwargs):
            captured.update(kwargs)
            return (
                {
                    "downscale_algorithm": "bilinear",
                    "upscale_algorithm": "bicubic",
                    "canvas_image": [torch.zeros((1, 8, 12, 3))],
                    "cropped_mask_for_blend": [blend_mask],
                    "cropped_to_canvas_x": [3],
                    "cropped_to_canvas_y": [2],
                    "cropped_to_canvas_w": [4],
                    "cropped_to_canvas_h": [4],
                    "canvas_to_orig_x": [1],
                    "canvas_to_orig_y": [1],
                    "canvas_to_orig_w": [10],
                    "canvas_to_orig_h": [6],
                },
                FakeCroppedImage(),
                "cropped_sampling_mask",
            )

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": FakeCropNode}),
    )

    output_mask = prepare_inpaint_output_mask(
        normalize_inpaint_config(
            image=SourceImage(),
            mask=SourceMask(),
            mask_grow_percent=20.0,
            mask_feather=32,
        )
    )

    expected = torch.zeros((1, 6, 10))
    expected[:, 1:5, 2:6] = 0.25
    assert torch.equal(output_mask, expected)
    assert captured["mask_expand_pixels"] == 2
    assert captured["mask_blend_pixels"] == 32
    assert captured["device_mode"] == "cpu (compatible)"


def test_stitcher_blend_mask_concatenates_batch_masks():
    torch = pytest.importorskip("torch")
    mask_a = torch.zeros((1, 3, 4))
    mask_b = torch.ones((1, 3, 4))

    output = stitcher_blend_mask({"cropped_mask_for_blend": [mask_a, mask_b]})

    assert output.shape == (2, 3, 4)
    assert torch.equal(output[0], mask_a[0])
    assert torch.equal(output[1], mask_b[0])


def test_prepare_inpaint_source_falls_back_to_full_frame_mask(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, 1:3, 1:3] = 1.0

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    source = prepare_inpaint_source(
        config=normalize_inpaint_config(image=image, mask=mask, mask_grow_percent=100.0),
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
    assert int(source.noise_mask.sum().item()) > 4


def test_prepare_inpaint_source_downscales_no_crop_large_full_frame(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 2048, 2048, 3))
    mask = torch.zeros((1, 2048, 2048))
    mask[:, 800:1200, 800:1200] = 1.0

    fake_comfy = ModuleType("comfy")
    fake_utils = ModuleType("comfy.utils")

    def fake_common_upscale(samples, width, height, upscale_method, crop):
        return torch.zeros((samples.shape[0], samples.shape[1], int(height), int(width)))

    fake_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    source = prepare_inpaint_source(
        config=normalize_inpaint_config(image=image, mask=mask),
        width=2048,
        height=2048,
    )

    assert source.used_crop is False
    assert source.working_dimensions(fallback_width=1, fallback_height=1) == (1024, 1024)
    assert source.image.shape == (1, 1024, 1024, 3)
    assert source.mask.shape == (1, 1024, 1024)


def test_prepare_inpaint_output_mask_keeps_no_crop_source_size(monkeypatch):
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 1024, 1024, 3))
    mask = torch.zeros((1, 1024, 1024))
    mask[:, 384:640, 384:640] = 1.0

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    output_mask = prepare_inpaint_output_mask(
        normalize_inpaint_config(
            image=image,
            mask=mask,
            max_full_frame_megapixels=0.25,
            max_full_frame_side=512,
        )
    )

    assert output_mask.shape == (1, 1024, 1024)


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


def test_apply_inpaint_color_match_returns_target_when_disabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    image = apply_inpaint_color_match(
        target_image="target",
        reference_image="reference",
        exclude_mask="mask",
        strength=0.0,
    )

    assert image == "target"


def test_apply_inpaint_color_match_uses_acly_node(monkeypatch):
    captured = {}

    class FakeColorMatch:
        @classmethod
        def execute(cls, **kwargs):
            captured.update(kwargs)
            return ("matched_image",)

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"INPAINT_ColorMatch": FakeColorMatch}),
    )

    image = apply_inpaint_color_match(
        target_image="decoded_target",
        reference_image="source_reference",
        exclude_mask="sampling_mask",
        strength=0.25,
    )

    assert image == "matched_image"
    assert captured == {
        "target": "decoded_target",
        "reference": "source_reference",
        "exclude_mask": "sampling_mask",
        "strength": 0.25,
    }


def test_apply_inpaint_color_match_requires_acly_node(monkeypatch):
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))

    with pytest.raises(ValueError, match="Acly/comfyui-inpaint-nodes"):
        apply_inpaint_color_match(
            target_image="decoded_target",
            reference_image="source_reference",
            exclude_mask="sampling_mask",
            strength=0.25,
        )
