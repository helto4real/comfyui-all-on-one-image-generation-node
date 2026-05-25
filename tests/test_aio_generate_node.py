import pytest

from nodes.aio_generate import AIOImageGenerate


class FakeImage:
    shape = (1, 768, 512, 3)


def test_main_node_exposes_core_inputs():
    inputs = AIOImageGenerate.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert "model_type" in required
    assert "weight_format" not in required
    assert "positive_prompt" in required
    assert "negative_prompt" in required
    assert "size mode" in required
    assert "max side" in required
    assert "aspect ratio" in required
    assert "width" not in required
    assert "height" not in required
    assert "model_settings" in optional
    assert "lora_config" in optional
    assert "image 1" in optional
    assert "image 2" in optional
    assert "image 3" in optional
    assert "image 4" in optional
    assert "reference_image" not in optional


def test_main_node_lists_gguf_text_encoder_category(monkeypatch):
    from nodes import aio_generate

    def filename_list(category):
        if category == "clip_gguf":
            return ["t5-q4.gguf"]
        return []

    monkeypatch.setattr(aio_generate, "_filename_list", filename_list)

    text_encoder_options = AIOImageGenerate.INPUT_TYPES()["required"]["text_encoder"][0]

    assert text_encoder_options == ["clip_gguf/t5-q4.gguf"]


def test_main_node_rejects_settings_family_mismatch():
    node = AIOImageGenerate()

    with pytest.raises(ValueError, match="Selected settings are for flux2_klein_9b"):
        node.generate(
            model_type="z_image_turbo",
            diffusion_model="model.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            positive_prompt="prompt",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=0,
            steps=0,
            cfg=0.0,
            sampler="auto",
            scheduler="auto",
            model_settings={"family": "flux2_klein_9b"},
        )


def test_main_node_passes_lora_config_to_adapter(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            return "image", {"samples": "latent"}

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        lora_config={"lora_1": {"on": True, "lora": "style", "strength": 0.8}},
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert captured["lora_config"]["loras"][0]["name"] == "style"
    assert '"loras": [{"enabled": true, "name": "style"' in run_info


def test_main_node_normalizes_named_reference_images(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs["reference_inputs"]
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs["reference_inputs"]
            return "image", {"samples": "latent"}

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        mask="mask",
        **{"image 1": "first", "image 2": "second"},
    )

    assert captured["validated"].images == ("first", "second")
    assert captured["generated"].mask == "mask"


def test_main_node_resolves_aspect_ratio_output_size(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "family": "flux2_klein_9b",
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info = AIOImageGenerate().generate(
        model_type="flux2_klein_9b",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        **{"size mode": "use aspect ratio", "max side": 1024, "aspect ratio": "16:9"},
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert captured["validated"]["width"] == 1024
    assert captured["validated"]["height"] == 576
    assert captured["generated"]["width"] == 1024
    assert captured["generated"]["height"] == 576
    assert '"height": 576' in run_info


def test_main_node_can_use_image_1_size(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            return "image", {"samples": "latent"}

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info = AIOImageGenerate().generate(
        model_type="flux2_klein_9b",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        **{
            "size mode": "use image 1 size",
            "max side": 1024,
            "aspect ratio": "1:1",
            "image 1": FakeImage(),
        },
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert captured["width"] == 512
    assert captured["height"] == 768
    assert '"size_mode": "use image 1 size"' in run_info


def test_main_node_keeps_legacy_width_height_compatibility(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            return "image", {"samples": "latent"}

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=768,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert captured["width"] == 1024
    assert captured["height"] == 768
