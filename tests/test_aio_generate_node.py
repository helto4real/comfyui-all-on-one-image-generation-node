import json
import math
from pathlib import Path

import pytest

from nodes.aio_generate import (
    AIOImageGenerate,
    AIO_GENERATE_SERIALIZED_WIDGET_NAMES,
    IMAGE_ORIGINAL_OUTPUT_INDEX,
    _validate_batch_count,
    image_output_is_required,
    output_is_reachable,
    workflow_output_has_link,
)
from nodes.info import AIOInpaintInfo, AIOModelInfo, AIOPIDInfo
from nodes.inpaint import AIOInpaint
from services import pipeline, privacy


ROOT = Path(__file__).resolve().parents[1]


class FakeImage:
    shape = (1, 768, 512, 3)


class FakeGeneratedImage:
    shape = (1, 769, 513, 3)


class FakeMask:
    shape = (1, 768, 512)


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
    assert required["use_zero_negative_conditioning"][1]["default"] is True
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
    assert required["batch_count"][1]["default"] == 1
    assert required["batch_count"][1]["min"] == 1
    assert required["batch_count"][1]["max"] == 64
    assert required["batch_count"][1]["step"] == 1
    required_names = list(required)
    assert required_names[required_names.index("use_zero_negative_conditioning") - 1] == "negative_prompt"
    assert required_names[required_names.index("use_zero_negative_conditioning") + 1] == "privacy_mode"
    assert required_names[required_names.index("batch_count") - 1] == "seed"
    assert required_names[required_names.index("batch_count") + 1] == "steps"
    assert required["second_pass_enabled"][1]["default"] is False
    assert required["second_pass_steps"][1]["default"] == 0
    assert required["second_pass_steps"][1]["min"] == 0
    assert required["second_pass_steps"][1]["max"] == 100
    assert required["second_pass_steps"][1]["step"] == 1
    assert required["second_pass_denoise"][1]["default"] == 0.15
    assert required["second_pass_denoise"][1]["min"] == 0.0
    assert required["second_pass_denoise"][1]["max"] == 1.0
    assert required["second_pass_upscale_ratio"][1]["default"] == 1.5
    assert required["second_pass_upscale_ratio"][1]["min"] == 1.0
    assert required["second_pass_upscale_ratio"][1]["max"] == 8.0
    assert required["second_pass_upscale_method"][0] == ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    assert required["second_pass_upscale_method"][1]["default"] == "lanczos"
    assert AIO_GENERATE_SERIALIZED_WIDGET_NAMES[-5:] == (
        "second_pass_enabled",
        "second_pass_steps",
        "second_pass_denoise",
        "second_pass_upscale_ratio",
        "second_pass_upscale_method",
    )
    assert AIO_GENERATE_SERIALIZED_WIDGET_NAMES[
        AIO_GENERATE_SERIALIZED_WIDGET_NAMES.index("use_zero_negative_conditioning") - 1
    ] == "negative_prompt"
    assert AIO_GENERATE_SERIALIZED_WIDGET_NAMES[
        AIO_GENERATE_SERIALIZED_WIDGET_NAMES.index("use_zero_negative_conditioning") + 1
    ] == "privacy_mode"
    assert AIO_GENERATE_SERIALIZED_WIDGET_NAMES[
        AIO_GENERATE_SERIALIZED_WIDGET_NAMES.index("batch_count") - 1
    ] == "seed"
    assert AIO_GENERATE_SERIALIZED_WIDGET_NAMES[
        AIO_GENERATE_SERIALIZED_WIDGET_NAMES.index("batch_count") + 1
    ] == "steps"
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
        "AIO_MODEL_INFO",
        "AIO_PID_INFO",
        "INT",
        "INT",
        "AIO_INPAINT_INFO",
        "IMAGE",
    )
    assert AIOImageGenerate.RETURN_NAMES == (
        "image",
        "latent",
        "run_info",
        "model_info",
        "pid_info",
        "width",
        "height",
        "inpaint_info",
        "image_original",
    )
    assert IMAGE_ORIGINAL_OUTPUT_INDEX == 8


def test_main_node_is_changed_uses_native_cache_for_public_inputs(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_external_cache_providers_registered", lambda: True)

    assert AIOImageGenerate.IS_CHANGED(privacy_mode=False) is False
    assert AIOImageGenerate.IS_CHANGED(model_settings={"privacy_mode": False}) is False


def test_main_node_is_changed_allows_private_ram_cache_without_external_provider(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_external_cache_providers_registered", lambda: False)

    assert AIOImageGenerate.IS_CHANGED(privacy_mode=True) is False


def test_main_node_is_changed_disables_private_cache_with_external_provider(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_external_cache_providers_registered", lambda: True)

    assert math.isnan(AIOImageGenerate.IS_CHANGED(privacy_mode=True))


def test_main_node_is_changed_disables_direct_private_model_settings_with_external_provider(monkeypatch):
    from nodes import aio_generate

    monkeypatch.setattr(aio_generate, "_external_cache_providers_registered", lambda: True)

    assert math.isnan(AIOImageGenerate.IS_CHANGED(model_settings={"privacy_mode": True}))
    assert math.isnan(AIOImageGenerate.IS_CHANGED(model_settings={"prompt_builder_privacy_mode": True}))


def test_info_nodes_expose_bundle_extractors():
    assert AIOModelInfo.RETURN_TYPES == ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE")
    assert AIOModelInfo.RETURN_NAMES == ("model", "clip", "positive", "negative", "vae")
    assert "model_info" in AIOModelInfo.INPUT_TYPES()["required"]
    assert AIOModelInfo().extract(
        {
            "model": "model",
            "clip": "clip",
            "positive": "positive",
            "negative": "negative",
            "vae": "vae",
        }
    ) == ("model", "clip", "positive", "negative", "vae")

    assert AIOPIDInfo.RETURN_TYPES == ("LATENT", "FLOAT", "INT")
    assert AIOPIDInfo.RETURN_NAMES == ("latent", "sigma", "step")
    assert "pid_info" in AIOPIDInfo.INPUT_TYPES()["required"]
    assert AIOPIDInfo().extract({"latent": "latent", "sigma": 0.5, "step": 3}) == ("latent", 0.5, 3)

    assert AIOInpaintInfo.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK")
    assert AIOInpaintInfo.RETURN_NAMES == ("source", "sample", "mask")
    assert "inpaint_info" in AIOInpaintInfo.INPUT_TYPES()["required"]
    assert AIOInpaintInfo().extract({"source": "source", "sample": "sample", "mask": "mask"}) == (
        "source",
        "sample",
        "mask",
    )


def test_aio_seed_frontend_randomizes_live_seed_before_queue():
    source = (ROOT / "web/js/aio_image_generate.js").read_text(encoding="utf-8")

    assert "const SEED_MAX = Number.MAX_SAFE_INTEGER;" in source
    assert "function randomizeAioSeedsBeforeQueue()" in source
    assert 'liveSeedControlMode(node) !== "randomize"' in source
    assert "writeAioSeedValue(node, seed)" in source
    assert "suspendSeedControlCallbacks(controlWidget)" in source
    assert "restoreQueuedAioSeeds(queuedSeeds)" in source
    assert "app.queuePrompt = wrappedQueuePrompt" in source
    assert "scheduleAioSeedQueuePatch(\"setup\")" in source
    assert "AIOSeedProbe" not in source


def test_aio_frontend_clears_standard_progress_text_after_execution():
    source = (ROOT / "web/js/aio_image_generate.js").read_text(encoding="utf-8")

    assert 'const PROGRESS_TEXT_WIDGET_NAME = "$$node-text-preview";' in source
    assert "function removeAioGenerateProgressTextWidget" in source
    assert "function clearAioGenerateRuntimePhases" in source
    assert "widget?.onRemove?.()" in source
    assert "delete node[AIO_RUNTIME_PHASE_NODE_KEY]" in source
    assert 'api.addEventListener?.("execution_success", scheduleAioGenerateProgressTextCleanup)' in source
    assert 'api.addEventListener?.("execution_error", scheduleAioGenerateProgressTextCleanup)' in source
    assert 'api.addEventListener?.("execution_interrupted", scheduleAioGenerateProgressTextCleanup)' in source
    assert "installAioGenerateProgressTextCleanup();" in source


def test_aio_frontend_tracks_runtime_phase_from_progress_text():
    source = (ROOT / "web/js/aio_image_generate.js").read_text(encoding="utf-8")

    assert 'const AIO_RUNTIME_PHASE_BRIDGE_KEY = "__aioGenerateRuntimePhaseBridgeInstalled";' in source
    assert "function handleAioGenerateProgressText" in source
    assert "const node = findAioGenerateNodeById(nodeId);" in source
    assert "setAioGenerateRuntimePhase(node, text);" in source
    assert 'api.addEventListener?.("progress_text", handleAioGenerateProgressText)' in source
    assert "installAioGenerateRuntimePhaseBridge();" in source

    set_phase_start = source.index("function setAioGenerateRuntimePhase")
    set_phase_end = source.index("function clearAioGenerateRuntimePhase", set_phase_start)
    set_phase_block = source[set_phase_start:set_phase_end]
    assert "!isAioGenerateNode(node)" in set_phase_block
    assert "node[AIO_RUNTIME_PHASE_NODE_KEY] = phase" in set_phase_block
    assert "scheduleAioGenerateRuntimePhaseDomUpdate(node)" in set_phase_block

    draw_start = source.index("function drawAioGenerateRuntimePhase")
    draw_end = source.index("function handleAioGenerateProgressText", draw_start)
    draw_block = source[draw_start:draw_end]
    assert "fitString(ctx, phase" in draw_block
    assert "ctx.roundRect" in draw_block


def test_aio_fixed_seed_button_sets_control_back_to_fixed():
    source = (ROOT / "web/js/aio_image_generate.js").read_text(encoding="utf-8")
    start = source.index("function ensureAioGenerateSeedButton")
    end = source.index("function defaultGraph", start)
    block = source[start:end]

    assert "button = node.addWidget" in block
    assert "writeAioSeedValue(node, seed)" in block
    assert 'writeAioSeedControlMode(node, "fixed")' in block
    assert "markNodeDirty(node)" in block


def test_aio_frontend_inserts_missing_batch_count_after_seed_on_restore():
    source = (ROOT / "web/js/aio_image_generate.js").read_text(encoding="utf-8")

    assert 'const BATCH_COUNT_WIDGET_NAME = "batch_count";' in source
    assert 'const USE_ZERO_NEGATIVE_CONDITIONING_WIDGET_NAME = "use_zero_negative_conditioning";' in source
    assert "function valuesWithMissingZeroNegativeConditioningSlot" in source
    assert "zeroNegativeIndex !== negativePromptIndex + 1" in source
    assert "values.length !== count - 1 && values.length !== count - 2" in source
    assert "normalized.splice(zeroNegativeIndex, 0, true)" in source
    assert "function valuesWithMissingBatchCountSlot" in source
    assert "batchIndex !== seedIndex + 1" in source
    assert "values.length !== count - 1" in source
    assert "normalized.splice(batchIndex, 0, 1)" in source
    assert "function normalizedAioGenerateWidgetValues" in source
    assert "valuesWithoutSeedButtonSlot(node, values)" in source
    assert "valuesWithMissingZeroNegativeConditioningSlot(node, withoutSeedButton)" in source
    assert "valuesWithMissingBatchCountSlot(node, withZeroNegativeConditioning)" in source
    assert "configureInfoWithNormalizedGenerateWidgets" in source


def test_aio_batch_count_validation_rejects_non_integer_and_out_of_range_values():
    assert _validate_batch_count(1) == 1
    assert _validate_batch_count(64) == 64

    for value in (0, 65, 1.5, "2", True):
        with pytest.raises(ValueError, match="batch_count"):
            _validate_batch_count(value)


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
        "2": {
            "class_type": "AIOModelInfo",
            "inputs": {"model_info": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "VAEConsumer", "inputs": {"vae": ["2", aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX]}},
    }

    assert output_is_reachable(prompt, "1", 0) is False
    assert output_is_reachable(prompt, "1", 1) is False
    assert aio_generate.info_output_is_reachable(
        prompt,
        None,
        "1",
        aio_generate.MODEL_INFO_OUTPUT_INDEX,
        "AIOModelInfo",
        aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX,
    ) is True


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
            return pipeline.GenerationResult(
                image="image",
                latent={"samples": "latent"},
                positive="positive",
                negative="negative",
                vae="vae",
                model=kwargs["loaded_model"],
                clip=kwargs["loaded_clip"],
            )

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert model_info["vae"] == "vae"
    assert model_info["model"] == "patched_model"
    assert model_info["clip"] == "patched_clip"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
    assert captured["lora_config"]["loras"][0]["name"] == "style"
    assert captured["loaded_model"] == "patched_model"
    assert captured["loaded_clip"] == "patched_clip"
    parsed = json.loads(run_info)
    assert captured["settings"]["use_zero_negative_conditioning"] is True
    assert parsed["settings"]["use_zero_negative_conditioning"] is True
    assert parsed["debug"]["prompts"]["use_zero_negative_conditioning"] is True
    assert parsed["loras"][0]["name"] == "style"
    assert (output_width, output_height) == (1024, 1024)


def test_main_node_drops_unrequested_model_info_model_and_clip(monkeypatch):
    from nodes import aio_generate

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
            return pipeline.GenerationResult(
                image="image",
                latent={"samples": "latent"},
                positive="positive",
                negative="negative",
                vae="vae",
                model="generated_model",
                clip="generated_clip",
            )

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(
        aio_generate,
        "_class_is_output_node",
        lambda class_type: class_type == "PreviewImage",
    )

    result = AIOImageGenerate().generate(
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

    model_info = result[3]
    assert model_info["model"] is None
    assert model_info["clip"] is None
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"


@pytest.mark.parametrize(
    ("output_index", "returned_key", "dropped_key"),
    [
        (0, "model", "clip"),
        (1, "clip", "model"),
    ],
)
def test_main_node_keeps_requested_model_info_model_or_clip(
    monkeypatch,
    output_index,
    returned_key,
    dropped_key,
):
    from nodes import aio_generate

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
            return pipeline.GenerationResult(
                image="image",
                latent={"samples": "latent"},
                positive="positive",
                negative="negative",
                vae="vae",
                model="generated_model",
                clip="generated_clip",
            )

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {
            "class_type": "AIOModelInfo",
            "inputs": {"model_info": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "BundleConsumer", "inputs": {"value": ["2", output_index]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(
        aio_generate,
        "_class_is_output_node",
        lambda class_type: class_type == "BundleConsumer",
    )

    result = AIOImageGenerate().generate(
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

    model_info = result[3]
    assert model_info[returned_key] == f"generated_{returned_key}"
    assert model_info[dropped_key] is None


def test_main_node_keeps_model_info_for_unknown_direct_bundle_consumer(monkeypatch):
    from nodes import aio_generate

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
            return pipeline.GenerationResult(
                image="image",
                latent={"samples": "latent"},
                positive="positive",
                negative="negative",
                vae="vae",
                model="generated_model",
                clip="generated_clip",
            )

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {
            "class_type": "CustomBundleConsumer",
            "inputs": {"value": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(
        aio_generate,
        "_class_is_output_node",
        lambda class_type: class_type == "CustomBundleConsumer",
    )

    result = AIOImageGenerate().generate(
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

    model_info = result[3]
    assert model_info["model"] == "generated_model"
    assert model_info["clip"] == "generated_clip"


def test_main_node_can_disable_zero_negative_conditioning(monkeypatch):
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

    _image, _latent, run_info, *_ = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="avoid blur",
        use_zero_negative_conditioning=False,
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    parsed = json.loads(run_info)
    assert captured["validated"]["settings"]["use_zero_negative_conditioning"] is False
    assert captured["generated"]["settings"]["use_zero_negative_conditioning"] is False
    assert parsed["settings"]["use_zero_negative_conditioning"] is False
    assert parsed["debug"]["prompts"]["use_zero_negative_conditioning"] is False


def test_main_node_passes_second_pass_config_and_strips_latent_sidecars(monkeypatch):
    from nodes import aio_generate

    captured = {}
    second_pass_info = {
        "enabled": True,
        "applied": True,
        "denoise": 0.15,
        "steps_input": 12,
        "steps": 12,
        "upscale_ratio": 1.5,
        "upscale_method": "lanczos",
        "first_pass_size": {"width": 1024, "height": 1024},
        "final_size": {"width": 1536, "height": 1536},
    }

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
            return (
                None,
                {
                    "samples": "second_pass_latent",
                    pipeline.SECOND_PASS_INFO_KEY: second_pass_info,
                    pipeline.SECOND_PASS_ORIGINAL_IMAGE_KEY: "first_pass_image",
                },
                "positive",
                "negative",
                "vae",
            )

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    (
        image,
        latent,
        run_info,
        _model_info,
        _pid_info,
        output_width,
        output_height,
        _inpaint_info,
        image_original,
    ) = AIOImageGenerate().generate(
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
        second_pass_enabled=True,
        second_pass_steps=12,
    )

    assert image is None
    assert latent == {"samples": "second_pass_latent"}
    assert image_original == "first_pass_image"
    assert captured["decode_image"] is True
    assert captured["second_pass_config"] == {
        "enabled": True,
        "denoise": 0.15,
        "steps_input": 12,
        "upscale_ratio": 1.5,
        "upscale_method": "lanczos",
        "decode_image": True,
        "return_image_original": True,
    }
    assert (output_width, output_height) == (1536, 1536)
    parsed = json.loads(run_info)
    assert parsed["width"] == 1536
    assert parsed["height"] == 1536
    assert parsed["second_pass"] == second_pass_info


def test_main_node_returns_original_image_output_when_second_pass_is_disabled(monkeypatch):
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
            return "first_pass_image", {"samples": "latent"}, "positive", "negative", "vae"

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", aio_generate.IMAGE_ORIGINAL_OUTPUT_INDEX]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    *_, image_original = AIOImageGenerate().generate(
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

    assert image_original == "first_pass_image"
    assert captured["decode_image"] is True
    assert captured["second_pass_config"]["enabled"] is False
    assert captured["second_pass_config"]["steps_input"] == 0


def test_main_node_passes_inpaint_config_and_uses_source_dimensions(monkeypatch):
    from nodes import aio_generate

    captured = {}
    inpaint_config = _inpaint_config(monkeypatch, mask_grow_percent=12.5, mask_feather=24, denoise=0.75, steps=6)

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

    _, _, run_info, _, _, output_width, output_height, _, _ = AIOImageGenerate().generate(
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
    assert captured["generated"]["inpaint_config"]["steps"] == 6
    assert (output_width, output_height) == (512, 768)
    parsed = json.loads(run_info)
    assert parsed["width"] == 512
    assert parsed["height"] == 768
    assert parsed["settings"]["size_mode"] == "use inpaint image size"
    assert parsed["debug"]["sampling"]["effective_sampling_steps"] == 6
    assert parsed["debug"]["inpaint"]["steps_input"] == 6
    assert parsed["debug"]["inpaint"]["effective_steps"] == 6


def test_main_node_returns_connected_inpaint_debug_previews(monkeypatch):
    from nodes import aio_generate

    captured = {}
    inpaint_config = _inpaint_config(monkeypatch)

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 20,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "ideogram4",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            previews = kwargs["inpaint_previews"]
            previews[pipeline.INPAINT_PREVIEW_SOURCE] = "debug_source"
            previews[pipeline.INPAINT_PREVIEW_SAMPLE] = "debug_sample"
            previews[pipeline.INPAINT_PREVIEW_MASK] = "debug_mask"
            return None, {"samples": "latent"}, "positive", "negative", None

    prompt = {
        "1": {"class_type": "AIOImageGenerate", "inputs": {}},
        "2": {
            "class_type": "AIOInpaintInfo",
            "inputs": {"inpaint_info": ["1", aio_generate.INPAINT_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", aio_generate.INPAINT_INFO_SOURCE_OUTPUT_INDEX]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["2", aio_generate.INPAINT_INFO_SAMPLE_OUTPUT_INDEX]}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["2", aio_generate.INPAINT_INFO_MASK_OUTPUT_INDEX]}},
    }

    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    (
        image,
        _latent,
        _run_info,
        _model_info,
        _pid_info,
        _output_width,
        _output_height,
        inpaint_info,
        _image_original,
    ) = AIOImageGenerate().generate(
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
            "unconditional_model": "unconditional.safetensors",
        },
        unique_id="1",
        prompt=prompt,
    )

    requested = captured["inpaint_previews"][pipeline.INPAINT_PREVIEW_REQUESTED]
    assert image is None
    assert captured["decode_image"] is False
    assert requested[pipeline.INPAINT_PREVIEW_SOURCE] is True
    assert requested[pipeline.INPAINT_PREVIEW_SAMPLE] is True
    assert requested[pipeline.INPAINT_PREVIEW_MASK] is True
    assert inpaint_info == {"source": "debug_source", "sample": "debug_sample", "mask": "debug_mask"}


def test_main_node_requests_all_inpaint_previews_when_workflow_field_links_are_unknown(monkeypatch):
    from nodes import aio_generate

    captured = {}
    inpaint_config = _inpaint_config(monkeypatch)

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 20,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "ideogram4",
            }

        def validate_inputs(self, **kwargs):
            return []

        def generate(self, **kwargs):
            captured.update(kwargs)
            return "image", {"samples": "latent"}, "positive", "negative", None

    extra_pnginfo = {
        "workflow": {
            "nodes": [
                {
                    "id": 1,
                    "outputs": [
                        {"name": "image", "links": None},
                        {"name": "latent", "links": None},
                        {"name": "run_info", "links": None},
                        {"name": "model_info", "links": None},
                        {"name": "pid_info", "links": None},
                        {"name": "width", "links": None},
                        {"name": "height", "links": None},
                        {"name": "inpaint_info", "links": [20]},
                        {"name": "image_original", "links": None},
                    ],
                },
                {"id": 2, "type": "AIOInpaintInfo"},
            ],
            "links": [
                [20, 1, aio_generate.INPAINT_INFO_OUTPUT_INDEX, 2, 0, "AIO_INPAINT_INFO"],
            ],
        },
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())

    AIOImageGenerate().generate(
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
            "unconditional_model": "unconditional.safetensors",
        },
        unique_id="1",
        prompt={"1": {"class_type": "AIOImageGenerate", "inputs": {}}},
        extra_pnginfo=extra_pnginfo,
    )

    requested = captured["inpaint_previews"][pipeline.INPAINT_PREVIEW_REQUESTED]
    assert requested[pipeline.INPAINT_PREVIEW_SOURCE] is True
    assert requested[pipeline.INPAINT_PREVIEW_SAMPLE] is True
    assert requested[pipeline.INPAINT_PREVIEW_MASK] is True


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

    _, _, run_info, _, _, output_width, output_height, _, _ = AIOImageGenerate().generate(
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

    _, _, run_info, _, _, output_width, output_height, _, _ = AIOImageGenerate().generate(
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


def test_krea2_inpaint_settings_prompt_overrides_main_prompt(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            resolved = dict(kwargs["model_settings"])
            resolved.update(
                {
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler": "auto",
                    "scheduler": "auto",
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

    _, _, run_info, *_ = AIOImageGenerate().generate(
        model_type="krea2",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="main prompt",
        negative_prompt="",
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        model_settings={
            "family": "krea2",
            "positive_prompt_override": "krea inpaint prompt",
            "positive_prompt_source": "krea2_inpaint_settings",
        },
        inpaint={"image": FakeImage(), "mask": FakeMask()},
    )

    assert captured["validated"]["positive_prompt"] == "krea inpaint prompt"
    assert captured["generated"]["positive_prompt"] == "krea inpaint prompt"
    parsed = json.loads(run_info)
    assert parsed["debug"]["prompts"]["effective_positive_prompt"] == "krea inpaint prompt"
    assert parsed["debug"]["prompts"]["positive_prompt_override_applied"] is True
    assert parsed["debug"]["prompts"]["positive_prompt_source"] == "krea2_inpaint_settings"


def test_krea2_inpaint_settings_prompt_is_ignored_without_inpaint(monkeypatch):
    from nodes import aio_generate

    captured = {}

    class FakeAdapter:
        version = "test"
        dimension_multiple = 16

        def resolve_settings(self, **kwargs):
            resolved = dict(kwargs["model_settings"])
            resolved.update(
                {
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler": "auto",
                    "scheduler": "auto",
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

    _, _, run_info, *_ = AIOImageGenerate().generate(
        model_type="krea2",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="main prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        model_settings={
            "family": "krea2",
            "positive_prompt_override": "krea inpaint prompt",
            "positive_prompt_source": "krea2_inpaint_settings",
        },
    )

    assert captured["validated"]["positive_prompt"] == "main prompt"
    assert captured["generated"]["positive_prompt"] == "main prompt"
    parsed = json.loads(run_info)
    assert parsed["debug"]["prompts"]["effective_positive_prompt"] == "main prompt"
    assert parsed["debug"]["prompts"]["positive_prompt_override_applied"] is False
    assert parsed["debug"]["prompts"]["positive_prompt_source"] == "node"


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_ideogram_prompt_builder_privacy_marker_redacts_settings_but_debug_exposes_prompt(monkeypatch, tmp_path):
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
    parsed = json.loads(run_info)
    encrypted = parsed["settings"]["positive_prompt_override"]
    assert privacy.is_encrypted_payload(encrypted)
    assert privacy.decrypt_text_if_encrypted(encrypted) == '{"secret":"private room"}'
    assert parsed["debug"]["prompts"]["effective_positive_prompt"] == '{"secret":"private room"}'


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

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert captured["decode_image"] is False
    assert captured["return_vae"] is False
    assert model_info["vae"] is None
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "2": {
            "class_type": "AIOPIDInfo",
            "inputs": {"pid_info": ["1", aio_generate.PID_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", aio_generate.PID_INFO_LATENT_OUTPUT_INDEX]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert pid_info["latent"] is captured_latent
    assert pid_info["sigma"] == 0.342
    assert pid_info["step"] == 6
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "2": {
            "class_type": "AIOPIDInfo",
            "inputs": {"pid_info": ["1", aio_generate.PID_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", aio_generate.PID_INFO_SIGMA_OUTPUT_INDEX]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "PreviewImage")

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert pid_info["latent"] is captured_latent
    assert pid_info["sigma"] == 0.123
    assert pid_info["step"] == 46
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "3": {
            "class_type": "AIOModelInfo",
            "inputs": {"model_info": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
        "4": {"class_type": "VAEConsumer", "inputs": {"vae": ["3", aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX]}},
    }
    output_nodes = {"SaveLatent", "VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert model_info["vae"] == "vae"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "4": {
            "class_type": "AIOModelInfo",
            "inputs": {"model_info": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
        "5": {"class_type": "VAEConsumer", "inputs": {"vae": ["4", aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX]}},
    }
    output_nodes = {"PreviewImage", "SaveLatent", "VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert model_info["vae"] == "vae"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "91": {"class_type": "AIOModelInfo", "inputs": {"model_info": ["83", aio_generate.MODEL_INFO_OUTPUT_INDEX]}},
        "92": {"class_type": "VAEConsumer", "inputs": {"vae": ["91", aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX]}},
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
                        {"name": "model_info", "links": [13]},
                        {"name": "pid_info", "links": None},
                        {"name": "width", "links": None},
                        {"name": "height", "links": None},
                        {"name": "inpaint_info", "links": None},
                        {"name": "image_original", "links": None},
                    ],
                },
                {
                    "id": 91,
                    "type": "AIOModelInfo",
                    "outputs": [
                        {"name": "model", "links": None},
                        {"name": "clip", "links": None},
                        {"name": "positive", "links": None},
                        {"name": "negative", "links": None},
                        {"name": "vae", "links": [14]},
                    ],
                },
            ],
            "links": [
                [10, 83, 0, 89, 0, "IMAGE"],
                [11, 83, 1, 90, 0, "LATENT"],
                [12, 83, 2, 93, 0, "STRING"],
                [13, 83, aio_generate.MODEL_INFO_OUTPUT_INDEX, 91, 0, "AIO_MODEL_INFO"],
                [14, 91, aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX, 92, 0, "VAE"],
            ],
        }
    }
    output_nodes = {"VAEConsumer"}
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type in output_nodes)

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert model_info["vae"] == "vae"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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
        "2": {
            "class_type": "AIOModelInfo",
            "inputs": {"model_info": ["1", aio_generate.MODEL_INFO_OUTPUT_INDEX]},
        },
        "3": {"class_type": "VAEConsumer", "inputs": {"vae": ["2", aio_generate.MODEL_INFO_VAE_OUTPUT_INDEX]}},
    }
    monkeypatch.setattr(aio_generate, "get_adapter", lambda model_type: FakeAdapter())
    monkeypatch.setattr(aio_generate, "_class_is_output_node", lambda class_type: class_type == "VAEConsumer")

    image, latent, run_info, model_info, pid_info, output_width, output_height, inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert model_info["vae"] == "vae"
    assert pid_info == {"latent": None, "sigma": 0.0, "step": 0}
    assert inpaint_info == {"source": None, "sample": None, "mask": None}
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

    image, latent, run_info, model_info, _pid_info, output_width, output_height, _inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
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

    image, latent, run_info, model_info, _pid_info, output_width, output_height, _inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
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

    image, latent, run_info, model_info, _pid_info, output_width, output_height, _inpaint_info, _image_original = AIOImageGenerate().generate(
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
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
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
