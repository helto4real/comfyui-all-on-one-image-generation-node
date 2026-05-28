import json

import pytest

from nodes.aio_generate import (
    AIOImageGenerate,
    image_output_is_required,
    output_is_reachable,
    workflow_output_has_link,
)


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
    assert "pid_enabled" in required
    assert required["pid_enabled"][1]["default"] is False
    assert "pid_save_vram" in required
    assert required["pid_save_vram"][1]["default"] is False
    assert "pid_diffusion_model" in required
    assert "pid_text_encoder" in required
    assert required["pid_vae"][0][0] == "pixel_space"
    assert required["pid_latent_format"][0] == ["flux", "sd3"]
    assert "size mode" in required
    assert "max side" in required
    assert required["max side"][1]["step"] == 1
    assert required["max side"][1]["min"] == 256
    assert required["max side"][1]["max"] == 4096
    assert "aspect ratio" in required
    assert required["multiple value"][0] == ["none", "8", "16", "32"]
    assert "width" not in required
    assert "height" not in required
    assert "model_settings" in optional
    assert "lora_config" in optional
    assert "model" in optional
    assert "clip" in optional
    assert "image 1" in optional
    assert "image 2" in optional
    assert "image 3" in optional
    assert "image 4" in optional
    assert "reference_image" not in optional
    assert AIOImageGenerate.RETURN_TYPES == (
        "IMAGE",
        "LATENT",
        "STRING",
        "CONDITIONING",
        "CONDITIONING",
        "VAE",
        "IMAGE",
    )
    assert AIOImageGenerate.RETURN_NAMES == (
        "image",
        "latent",
        "run_info",
        "positive",
        "negative",
        "vae",
        "image_pid",
    )


def test_image_output_is_required_for_connected_output_node(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")
    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }

    assert image_output_is_required(prompt, "1") is True


def test_image_output_is_not_required_for_latent_only_output(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "SaveLatent")
    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "SaveLatent", "inputs": {"samples": ["1", 1]}},
    }

    assert image_output_is_required(prompt, "1") is False
    assert output_is_reachable(prompt, "1", 1) is True
    assert output_is_reachable(prompt, "1", 5) is False


def test_unused_image_branch_does_not_force_decode(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "SaveLatent")
    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "ImageInvert", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "SaveLatent", "inputs": {"samples": ["1", 1]}},
    }

    assert image_output_is_required(prompt, "1") is False


def test_vae_output_reachability_uses_socket_index(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "VAEConsumer")
    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "VAEConsumer", "inputs": {"vae": ["1", 5]}},
    }

    assert output_is_reachable(prompt, "1", 0) is False
    assert output_is_reachable(prompt, "1", 1) is False
    assert output_is_reachable(prompt, "1", 5) is True


def test_workflow_output_has_link_reads_full_canvas_links():
    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {
                    "id": 83,
                    "outputs": [
                        {"name": "image", "links": [10]},
                        {"name": "latent", "links": [11]},
                        {"name": "run_info", "links": [12]},
                        {"name": "positive", "links": None},
                        {"name": "negative", "links": None},
                        {"name": "vae", "links": [13]},
                    ],
                }
            ],
            "links": [
                [11, 83, 1, 90, 0, "LATENT"],
                [13, 83, 5, 91, 0, "VAE"],
            ],
        }
    }

    assert workflow_output_has_link(extra_pnginfo, "83", 1) is True
    assert workflow_output_has_link(extra_pnginfo, "83", 3) is False
    assert workflow_output_has_link(extra_pnginfo, "83", 5) is True


def test_main_node_lists_gguf_text_encoder_category(monkeypatch):
    from nodes import aio_generate

    def filename_list(category):
        if category == "clip_gguf":
            return ["t5-q4.gguf"]
        return []

    monkeypatch.setattr(aio_generate, "_filename_list", filename_list)

    text_encoder_options = AIOImageGenerate.INPUT_TYPES()["required"]["text_encoder"][0]

    assert text_encoder_options == ["clip_gguf/t5-q4.gguf"]


def test_main_node_prefers_pid_defaults_when_present(monkeypatch):
    from nodes import aio_generate

    def filename_list(category):
        if category == "diffusion_models":
            return [
                "other.safetensors",
                "pid/pid_flux1_1024_to_4096_4step_bf16.safetensors",
            ]
        if category == "text_encoders":
            return [
                "other.safetensors",
                "pid/gemma_2_2b_it_elm_bf16.safetensors",
            ]
        if category == "vae":
            return ["ae.safetensors"]
        return []

    monkeypatch.setattr(aio_generate, "_filename_list", filename_list)
    required = AIOImageGenerate.INPUT_TYPES()["required"]

    assert required["pid_diffusion_model"][0][0] == (
        "diffusion_models/pid/pid_flux1_1024_to_4096_4step_bf16.safetensors"
    )
    assert required["pid_text_encoder"][0][0] == (
        "text_encoders/pid/gemma_2_2b_it_elm_bf16.safetensors"
    )
    assert required["pid_vae"][0][:2] == ["pixel_space", "vae/ae.safetensors"]


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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        model="patched_model",
        clip="patched_clip",
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert captured["lora_config"]["loras"][0]["name"] == "style"
    assert captured["loaded_model"] == "patched_model"
    assert captured["loaded_clip"] == "patched_clip"
    assert '"loras": [{"enabled": true, "name": "style"' in run_info


def test_main_node_skips_image_decode_for_latent_only_prompt(monkeypatch):
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
            return None, {"samples": "latent"}, "positive", "negative", None

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "SaveLatent", "inputs": {"samples": ["1", 1]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "SaveLatent")

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="1",
        prompt=prompt,
    )

    assert image is None
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert captured["decode_image"] is False
    assert captured["return_vae"] is False
    assert loaded_vae is None
    assert '"width": 1024' in run_info


def test_main_node_does_not_run_pid_when_pid_output_is_not_connected(monkeypatch):
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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    def fail_pid(**kwargs):
        raise AssertionError("PID should not run")

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate.pipeline, "generate_pid_upscale", fail_pid)

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        pid_enabled=True,
        pid_save_vram=True,
    )

    parsed = json.loads(run_info)
    assert image == "image"
    assert image_pid is None
    assert captured["decode_image"] is True
    assert parsed["pid"]["enabled"] is True
    assert parsed["pid"]["connected"] is False
    assert parsed["pid"]["ran"] is False


def test_main_node_pid_disabled_connected_returns_base_image(monkeypatch):
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
            return "base_image", {"samples": "latent"}, "positive", "negative", None

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 6]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="1",
        prompt=prompt,
        pid_enabled=False,
    )

    parsed = json.loads(run_info)
    assert image == "base_image"
    assert image_pid == "base_image"
    assert captured["decode_image"] is True
    assert parsed["pid"]["connected"] is True
    assert parsed["pid"]["ran"] is False


def test_main_node_pid_enabled_connected_uses_generated_latent_and_prompt(monkeypatch):
    from nodes import aio_generate

    captured_adapter = {}
    captured_pid = {}

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
            captured_adapter.update(kwargs)
            return None, {"samples": "generated_latent"}, "positive", "negative", None

    def fake_pid(**kwargs):
        captured_pid.update(kwargs)
        return "pid_image", {
            "input_size": 1024,
            "output_size": 4096,
            "target_width": 4096,
            "target_height": 4096,
            "pid_backbone": "flux1",
            "source_latent_channels": 16,
            "expected_latent_channels": 16,
            "selected_model_compatible": True,
            "validation": "passed",
        }

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 6]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate.pipeline, "generate_pid_upscale", fake_pid)
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt text",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=123,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        unique_id="1",
        prompt=prompt,
        pid_enabled=True,
        pid_save_vram=True,
        pid_diffusion_model="diffusion_models/pid/pid_flux1_1024_to_4096_4step_bf16.safetensors",
        pid_text_encoder="text_encoders/pid/gemma_2_2b_it_elm_bf16.safetensors",
        pid_vae="pixel_space",
        pid_latent_format="flux",
    )

    parsed = json.loads(run_info)
    assert image is None
    assert image_pid == "pid_image"
    assert captured_adapter["decode_image"] is False
    assert captured_pid["source_latent"] == {"samples": "generated_latent"}
    assert captured_pid["positive_prompt"] == "prompt text"
    assert captured_pid["seed"] == 123
    assert captured_pid["save_vram"] is True
    assert parsed["pid"]["ran"] is True
    assert parsed["pid"]["target_width"] == 4096
    assert parsed["pid"]["target_height"] == 4096
    assert parsed["pid"]["pid_backbone"] == "flux1"
    assert parsed["pid"]["source_latent_channels"] == 16
    assert parsed["pid"]["validation"] == "passed"


def test_main_node_returns_vae_for_valid_latent_only_vae_prompt(monkeypatch):
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
            return None, {"samples": "latent"}, "positive", "negative", "vae"

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "SaveLatent", "inputs": {"samples": ["1", 1]}},
        "3": {"class_type": "VAEConsumer", "inputs": {"vae": ["1", 5]}},
    }
    output_nodes = {"SaveLatent", "VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="1",
        prompt=prompt,
    )

    assert image is None
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert captured["decode_image"] is False
    assert captured["return_vae"] is True
    assert '"width": 1024' in run_info


def test_main_node_returns_vae_when_image_and_latent_are_connected(monkeypatch):
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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "SaveLatent", "inputs": {"samples": ["1", 1]}},
        "4": {"class_type": "VAEConsumer", "inputs": {"vae": ["1", 5]}},
    }
    output_nodes = {"PreviewImage", "SaveLatent", "VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="1",
        prompt=prompt,
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert captured["decode_image"] is True
    assert captured["return_vae"] is True
    assert '"width": 1024' in run_info


def test_main_node_uses_workflow_links_for_image_and_vae_outputs(monkeypatch):
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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    prompt = {
        "83": {"class_type": "AIOImageGenerate", "inputs": {}},
        "91": {"class_type": "VAEConsumer", "inputs": {"vae": ["83", 5]}},
    }
    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {
                    "id": 83,
                    "outputs": [
                        {"name": "image", "links": [10]},
                        {"name": "latent", "links": [11]},
                        {"name": "run_info", "links": [12]},
                        {"name": "positive", "links": None},
                        {"name": "negative", "links": None},
                        {"name": "vae", "links": [13]},
                    ],
                }
            ],
            "links": [
                [10, 83, 0, 89, 0, "IMAGE"],
                [11, 83, 1, 90, 0, "LATENT"],
                [12, 83, 2, 92, 0, "STRING"],
                [13, 83, 5, 91, 0, "VAE"],
            ],
        }
    }
    output_nodes = {"VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="83",
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert captured["decode_image"] is True
    assert captured["return_vae"] is True
    assert '"width": 1024' in run_info


def test_main_node_returns_vae_when_latent_is_not_connected(monkeypatch):
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
            return None, {"samples": "latent"}, "positive", "negative", "vae"

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "VAEConsumer", "inputs": {"vae": ["1", 5]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "VAEConsumer")

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        unique_id="1",
        prompt=prompt,
    )

    assert image is None
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert captured["decode_image"] is False
    assert captured["return_vae"] is True
    assert '"width": 1024' in run_info


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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
        **{
            "size mode": "use aspect ratio",
            "max side": 1024,
            "aspect ratio": "16:9",
            "multiple value": "16",
        },
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert captured["validated"]["width"] == 1024
    assert captured["validated"]["height"] == 576
    assert captured["generated"]["width"] == 1024
    assert captured["generated"]["height"] == 576
    assert captured["generated"]["settings"]["multiple_value"] == "16"
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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
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
            "multiple value": "16",
            "image 1": FakeImage(),
        },
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert captured["width"] == 512
    assert captured["height"] == 768
    assert '"size_mode": "use image 1 size"' in run_info
    assert '"multiple_value": "16"' in run_info


@pytest.mark.parametrize("model_type", ["z_image_turbo", "flux2_klein_9b"])
def test_main_node_uses_text_to_image_when_image_1_size_has_no_image(monkeypatch, model_type):
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
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info, positive, negative, loaded_vae, image_pid = AIOImageGenerate().generate(
        model_type=model_type,
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
            "aspect ratio": "16:9",
            "multiple value": "16",
        },
    )

    parsed = json.loads(run_info)
    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert captured["validated"]["reference_inputs"].count == 0
    assert captured["generated"]["reference_inputs"].count == 0
    assert captured["generated"]["width"] == 1024
    assert captured["generated"]["height"] == 576
    assert captured["generated"]["settings"]["size_mode"] == "use aspect ratio"
    assert parsed["settings"]["size_mode"] == "use aspect ratio"


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
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

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
