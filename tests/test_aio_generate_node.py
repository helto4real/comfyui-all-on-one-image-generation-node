import json

import pytest

from nodes.aio_generate import (
    AIOImageGenerate,
    image_output_is_required,
    output_is_reachable,
    workflow_output_has_link,
)
from nodes.inpaint import AIOInpaint
from services import privacy


class FakeImage:
    shape = (1, 768, 512, 3)


class FakeGeneratedImage:
    shape = (1, 769, 513, 3)


def _inpaint_config(monkeypatch, **kwargs):
    del monkeypatch
    torch = pytest.importorskip("torch")
    image = torch.rand((1, 768, 512, 3))
    mask = torch.zeros((1, 768, 512))
    mask[:, 256:512, 192:320] = 1.0
    return AIOInpaint().configure(image=image, mask=mask, **kwargs)[0]


def test_main_node_exposes_core_inputs():
    inputs = AIOImageGenerate.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert "model_type" in required
    assert "weight_format" not in required
    assert "positive_prompt" in required
    assert "negative_prompt" in required
    assert required["privacy_mode"][1]["default"] is False
    assert "pid_enabled" not in required
    assert "pid_save_vram" not in required
    assert "pid_capture_enabled" not in required
    assert required["pid_capture_step"][1]["default"] == 0
    assert "pid_degrade_sigma" not in required
    assert "pid_diffusion_model" not in required
    assert "pid_text_encoder" not in required
    assert "pid_vae" not in required
    assert "pid_latent_format" not in required
    assert "size mode" in required
    assert "max side" in required
    assert required["max side"][1]["step"] == 1
    assert required["max side"][1]["min"] == 256
    assert required["max side"][1]["max"] == 4096
    assert "aspect ratio" in required
    assert required["multiple value"][0] == ["none", "8", "16", "32"]
    assert required["seed"][1]["control_after_generate"] == "fixed"
    assert "width" not in required
    assert "height" not in required
    assert "model_settings" in optional
    assert "lora_config" in optional
    assert "inpaint" in optional
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
        "LATENT",
        "FLOAT",
        "INT",
        "INT",
    )
    assert AIOImageGenerate.RETURN_NAMES == (
        "image",
        "latent",
        "run_info",
        "positive",
        "negative",
        "vae",
        "pid_latent",
        "pid_sigma",
        "width",
        "height",
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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert json.loads(run_info)["loras"][0]["name"] == "style"
    assert (output_width, output_height) == (1024, 1024)


def test_main_node_passes_inpaint_config_and_uses_source_dimensions(monkeypatch):
    from nodes import aio_generate

    captured = {}
    inpaint_config = _inpaint_config(monkeypatch, mask_grow_percent=12.5, mask_feather=24, denoise=0.75)

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            captured["resolved"] = kwargs
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 20,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "ideogram4",
            }

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    _, _, run_info, _, _, _, _, _, output_width, output_height = AIOImageGenerate().generate(
        model_type="ideogram4",
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
        inpaint=inpaint_config,
        model_settings={
            "family": "ideogram4",
            "prompt_builder_width": 1088,
            "prompt_builder_height": 608,
        },
        **{
            "size mode": "use aspect ratio",
            "max side": 512,
            "aspect ratio": "1:1",
            "multiple value": "16",
        },
    )

    assert captured["resolved"]["width"] == 512
    assert captured["resolved"]["height"] == 768
    assert captured["validated"]["inpaint_config"]["mask_grow_percent"] == 12.5
    assert captured["generated"]["inpaint_config"]["mask_feather"] == 24
    assert captured["generated"]["inpaint_config"]["denoise"] == 0.75
    assert (output_width, output_height) == (512, 768)
    parsed = json.loads(run_info)
    assert parsed["width"] == 512
    assert parsed["height"] == 768
    assert parsed["settings"]["size_mode"] == "use inpaint image size"


def test_main_node_reports_actual_decoded_image_dimensions(monkeypatch):
    from nodes import aio_generate

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "width": 512,
                "height": 768,
                "steps": 4,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            return FakeGeneratedImage(), {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    _, _, run_info, _, _, _, _, _, output_width, output_height = AIOImageGenerate().generate(
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
            "size mode": "use aspect ratio",
            "max side": 512,
            "aspect ratio": "1:1",
            "multiple value": "16",
        },
    )

    assert (output_width, output_height) == (513, 769)
    parsed = json.loads(run_info)
    assert parsed["width"] == 513
    assert parsed["height"] == 769


def test_main_node_rejects_inpaint_for_unsupported_profile(monkeypatch):
    from adapters import z_image_turbo

    monkeypatch.setattr(z_image_turbo.gguf_backend, "is_available", lambda: True)
    inpaint_config = _inpaint_config(monkeypatch)

    with pytest.raises(ValueError, match="z_image_turbo does not currently support inpaint"):
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
            inpaint=inpaint_config,
        )


def test_main_node_decrypts_private_prompt_widgets(monkeypatch, tmp_path):
    from nodes import aio_generate
    from services import privacy

    monkeypatch.setattr(aio_generate.privacy, "config_dir", lambda: tmp_path)
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
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    positive = json.dumps(privacy.encrypt_state({"value": "private positive"}, base_dir=tmp_path))
    negative = json.dumps(privacy.encrypt_state({"value": "private negative"}, base_dir=tmp_path))
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt=positive,
        negative_prompt=negative,
        privacy_mode=True,
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert captured["validated"]["positive_prompt"] == "private positive"
    assert captured["validated"]["negative_prompt"] == "private negative"
    assert captured["generated"]["positive_prompt"] == "private positive"
    assert captured["generated"]["negative_prompt"] == "private negative"


def _workflow_node_with_prompt_widget(prompt_value):
    names = list(AIOImageGenerate.INPUT_TYPES()["required"])
    values = [""] * len(names)
    values[names.index("positive_prompt")] = prompt_value
    return {
        "id": 221,
        "type": "AIOImageGenerate",
        "inputs": [{"name": name, "widget": {"name": name}} for name in names],
        "widgets_values": values,
    }


def test_main_node_recovers_unlinked_prompt_from_workflow_widget_values(monkeypatch):
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
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        unique_id="221",
        prompt={"221": {"inputs": {"positive_prompt": ""}}},
        extra_pnginfo={"workflow": {"nodes": [_workflow_node_with_prompt_widget("visible prompt")]}},
    )

    assert captured["validated"]["positive_prompt"] == "visible prompt"
    assert captured["generated"]["positive_prompt"] == "visible prompt"


def test_main_node_does_not_recover_prompt_when_input_is_linked(monkeypatch):
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
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        unique_id="221",
        prompt={"221": {"inputs": {"positive_prompt": ["17", 0]}}},
        extra_pnginfo={"workflow": {"nodes": [_workflow_node_with_prompt_widget("visible prompt")]}},
    )

    assert captured["validated"]["positive_prompt"] == ""
    assert captured["generated"]["positive_prompt"] == ""


def test_ideogram_prompt_builder_overrides_prompt_and_dimensions(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"

        def resolve_settings(self, **kwargs):
            captured["resolved"] = kwargs
            resolved = dict(kwargs["model_settings"])
            resolved.update(
                {
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "steps": 8,
                    "cfg": 7.0,
                    "sampler": "euler",
                    "scheduler": "ideogram4",
                }
            )
            return resolved

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    _, _, run_info, _, _, _, _, _, output_width, output_height = AIOImageGenerate().generate(
        model_type="ideogram4",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="",
        negative_prompt="",
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        model_settings={
            "family": "ideogram4",
            "positive_prompt_override": '{"compositional_deconstruction":{"background":"Room","elements":[]}}',
            "positive_prompt_source": "ideogram4_prompt_builder",
            "prompt_builder_width": 1088,
            "prompt_builder_height": 608,
            "prompt_builder_max_side": 1088,
            "prompt_builder_aspect_ratio": "16:9",
            "prompt_builder_multiple_value": "16",
        },
        **{
            "size mode": "use aspect ratio",
            "max side": 512,
            "aspect ratio": "1:1",
            "multiple value": "16",
        },
    )

    assert captured["resolved"]["width"] == 1088
    assert captured["resolved"]["height"] == 608
    assert captured["validated"]["positive_prompt"] == '{"compositional_deconstruction":{"background":"Room","elements":[]}}'
    assert captured["generated"]["positive_prompt"] == '{"compositional_deconstruction":{"background":"Room","elements":[]}}'
    assert captured["generated"]["width"] == 1088
    assert captured["generated"]["height"] == 608
    assert (output_width, output_height) == (1088, 608)
    parsed = json.loads(run_info)
    assert parsed["width"] == 1088
    assert parsed["height"] == 608
    assert parsed["settings"]["positive_prompt_source"] == "ideogram4_prompt_builder"


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_ideogram_prompt_builder_privacy_marker_redacts_run_info(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy, "config_dir", lambda: tmp_path)

    captured = {}

    class FakeProfile:
        display_name = "Ideogram 4"

    class FakeAdapter:
        version = "0.1.0"

        def profile(self):
            return FakeProfile()

        def resolve_settings(self, **kwargs):
            resolved = dict(kwargs["model_settings"])
            resolved.update(
                {
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler": "euler",
                    "scheduler": "ideogram4",
                }
            )
            return resolved

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return "image", {"samples": "latent"}, "positive", "negative", "vae"

    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    encrypted_prompt = privacy.encrypt_state({"value": '{"secret":"private room"}'})

    _, _, run_info, *_ = AIOImageGenerate().generate(
        model_type="ideogram4",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="",
        negative_prompt="",
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        privacy_mode=False,
        model_settings={
            "family": "ideogram4",
            "positive_prompt_override": encrypted_prompt,
            "positive_prompt_source": "ideogram4_prompt_builder",
            "prompt_builder_privacy_mode": True,
        },
        **{
            "size mode": "use aspect ratio",
            "max side": 512,
            "aspect ratio": "1:1",
            "multiple value": "16",
        },
    )

    assert captured["generated"]["positive_prompt"] == '{"secret":"private room"}'
    assert "private room" not in run_info
    parsed = json.loads(run_info)
    encrypted = parsed["settings"]["positive_prompt_override"]
    assert privacy.is_encrypted_payload(encrypted)
    assert privacy.decrypt_text_if_encrypted(encrypted) == '{"secret":"private room"}'


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 1024)


def test_main_node_does_not_capture_pid_latent_when_pid_outputs_are_not_connected(monkeypatch):
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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    )

    assert image == "image"
    assert pid_latent is None
    assert pid_sigma == 0.0
    assert captured["decode_image"] is True
    assert captured["pid_capture_step"] is None
    assert "pid" not in json.loads(run_info)
    assert (output_width, output_height) == (1024, 1024)


def test_main_node_connected_pid_latent_output_captures_step(monkeypatch):
    from nodes import aio_generate

    captured = {}
    captured_latent = {"samples": "captured_latent", "pid_sigma": 0.342}

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
            return None, {
                "samples": "final_latent",
                aio_generate.pipeline.PID_CAPTURE_KEY: {
                    "latent": captured_latent,
                    "sigma": 0.342,
                    "step": kwargs["pid_capture_step"],
                },
            }, "positive", "negative", None

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 6]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
        pid_capture_step=6,
    )

    assert image is None
    assert pid_latent is captured_latent
    assert pid_sigma == 0.342
    assert captured["decode_image"] is False
    assert captured["pid_capture_step"] == 6
    assert "pid" not in json.loads(run_info)
    assert (output_width, output_height) == (1024, 1024)


def test_main_node_connected_pid_sigma_output_auto_selects_capture_step(monkeypatch):
    from nodes import aio_generate

    captured = {}
    captured_latent = {"samples": "captured_latent"}

    class FakeAdapter:
        version = "test"

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 50,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            return "image", {
                "samples": "final_latent",
                aio_generate.pipeline.PID_CAPTURE_KEY: {
                    "latent": captured_latent,
                    "sigma": 0.123,
                    "step": kwargs["pid_capture_step"],
                },
            }, "positive", "negative", None

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 7]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt text",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=123,
        steps=50,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        unique_id="1",
        prompt=prompt,
        pid_capture_step=0,
    )

    assert image == "image"
    assert pid_latent is captured_latent
    assert pid_sigma == 0.123
    assert captured["decode_image"] is False
    assert captured["pid_capture_step"] == 46
    assert (output_width, output_height) == (1024, 1024)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 1024)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 1024)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 1024)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 1024)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 576)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (512, 768)


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

    image, latent, run_info, positive, negative, loaded_vae, pid_latent, pid_sigma, output_width, output_height = AIOImageGenerate().generate(
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
    assert (output_width, output_height) == (1024, 576)


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
