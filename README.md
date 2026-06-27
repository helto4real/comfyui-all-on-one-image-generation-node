# AIO Image Generate

A classic ComfyUI custom node pack for an extensible all-in-one image generation node. The visible node is a simple facade, while model-family behavior lives in profiles, adapters, loader backends, validation services, and shared progress utilities.

## Installation

Clone or copy this folder into `ComfyUI/custom_nodes`, then restart ComfyUI. The pack has no extra runtime dependency beyond ComfyUI itself.

## Nodes

- `AIO Image Generate`
- `Z-Image Turbo Settings`
- `FLUX.2 Klein 9B Settings`
- `Ideogram 4 Prompt Builder`
- `Ideogram 4 Settings`
- `Krea 2 Settings`
- `AIO Inpaint`
- `AIO LoRA Configuration`
- `AIO Load Pipeline Models`

All nodes appear under `AIO/Image`.

## Supported Model Families

- `z_image_turbo`: text-to-image generation, defaults to 8 steps and CFG 1.0. Negative prompts are ignored by default and reported in `run_info.warnings`.
- `flux2_klein_9b`: text-to-image, reference-image, and AIO Inpaint generation, distilled defaults to 4 steps and CFG 1.0. Reference mode is inferred from how many reference images are connected. When `ComfyUI-Inpaint-CropAndStitch` is installed, Flux inpaint samples a cropped working area and stitches the decoded result back to the original canvas size.
- `ideogram4`: local open-weight Ideogram 4 text-to-image generation, defaults to the official 20-step Ideogram scheduler preset with dual-model CFG 7.0. Negative prompts are ignored by default and reported in `run_info.warnings`.
- `krea2`: local Krea 2 text-to-image generation, defaults to the provided workflow's 8-step `er_sde` / `simple` sampler path, CFG 1.0, and 1344x2048 canvas. Negative prompts are ignored by default through zeroed positive conditioning.

## Supported Formats

- `safetensors`: native ComfyUI model-path resolution through `folder_paths`.
- `gguf`: optional only. GGUF requires a compatible external backend such as ComfyUI-GGUF. The node detects GGUF per selected file extension and does not silently fall back to safetensors.
- GGUF text encoders are listed through the installed backend's `clip_gguf` folder key when available. This matches ComfyUI-GGUF's `*CLIPLoader (GGUF)` pattern, which can list regular and GGUF text encoder files.

If any selected model file ends in `.gguf` without a compatible backend, the node raises:

```text
A GGUF model file was selected, but no compatible GGUF backend was detected.
```

## Model Folder Expectations

The main node resolves filenames lazily at execution time:

- diffusion model: `models/diffusion_models`, `models/unet`, `models/checkpoints`, or backend-provided GGUF keys such as `unet_gguf` / `model_gguf`
- text encoder: `models/text_encoders`, `models/clip`, or backend-provided GGUF key `clip_gguf`
- VAE: `models/vae`, plus backend-provided `vae_gguf` when available

Ideogram 4 expects the conditional diffusion model, unconditional diffusion model, Qwen3-VL text encoder, and FLUX.2 VAE from the Comfy-Org Ideogram 4 packaging. A typical setup is:

- conditional diffusion model: `ideogram4/ideogram4_fp8_scaled.safetensors`
- unconditional diffusion model: `diffusion_models/ideogram4/ideogram4_unconditional_fp8_scaled.safetensors`
- text encoder: `qwen3vl_8b_fp8_scaled.safetensors`
- VAE: `flux.2/flux2-vae.safetensors`

Krea 2 expects a Krea 2 diffusion model, a Krea 2 compatible Qwen3-VL text encoder, and the Qwen Image VAE. The workflow defaults are:

- diffusion model: `krea/krea2_turbo_fp8.safetensors`
- text encoder: `qwen3vl_4b_fp8_scaled.safetensors`
- VAE: `qwen_image_vae.safetensors`

The dropdown may prefix values with their category when multiple folders are searched.

## Basic Usage

1. Add `AIO Image Generate`.
2. Select `model_type`.
3. Select diffusion model, text encoder, and VAE files.
4. Enter a positive prompt.
5. Optionally attach the matching model-specific settings node.
6. For Ideogram 4, optionally connect `Ideogram 4 Prompt Builder` to `Ideogram 4 Settings` to use structured JSON prompting.
7. Optionally attach `AIO LoRA Configuration` to apply one or more LoRAs.
8. Optionally attach `AIO Inpaint` to edit only a masked source-image area on supported model families.
9. Optionally use `AIO Load Pipeline Models` and external `MODEL`/`CLIP` patch nodes after the LoRA phase.
10. Optionally enable the second sampler pass to upscale the first generated image, VAE-encode it, and refine it with low denoise.
11. Connect `IMAGE` to Preview Image or Save Image.

The node is not an output node, so it is safe for API-mode workflows.

## Settings Nodes

`Z-Image Turbo Settings` returns an `AIO_MODEL_SETTINGS` dict with speed preset, forced steps, prompt enhancement, negative-prompt policy, precision policy, attention backend, Torch compile, and performance-apply timing.

`FLUX.2 Klein 9B Settings` returns an `AIO_MODEL_SETTINGS` dict with distilled/base variant, guidance, reference strength, precision policy, memory policy, shift parameters, reference scaling controls, attention backend, Torch compile, and performance-apply timing. FLUX.2 Klein edit mode is inferred from connected reference image sockets.

`Ideogram 4 Settings` returns an `AIO_MODEL_SETTINGS` dict with the unconditional model toggle and model path, sampling preset, dual CFG, final CFG override window, AuraFlow sampling shift, precision policy, attention backend, Torch compile, and performance-apply timing. The official presets use Ideogram 4 sigmas; `Workflow Compatible` uses the saved workflow's simple scheduler path. Disable `run_unconditional_model` for turbo LoRA workflows that should skip the separate unconditional diffusion model and run the guider with the conditional model only.

`Krea 2 Settings` returns an `AIO_MODEL_SETTINGS` dict with conditioning rebalance controls, precision policy, attention backend, Torch compile, performance-apply timing, and CUDA fp16 accumulation callbacks. The default rebalance multiplier is `4.0`, with workflow layer weights `1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0`.

`Ideogram 4 Prompt Builder` returns an `AIO_IDEOGRAM4_PROMPT` payload plus convenience `prompt`, `preview`, `bboxes`, `width`, and `height` outputs. Connect its first output to `Ideogram 4 Settings`. When connected, the generated JSON prompt replaces the main node's `positive_prompt`, and the builder's resolved dimensions replace the main node's size controls for Ideogram 4 only. The builder uses the same `max side`, `aspect ratio`, and `multiple value` calculation as `AIO Image Generate`; it does not expose raw width/height inputs.

The prompt builder's JSON output is KJ-compatible: compact output uses the same key order, bbox normalization, palette casing, and compact separators as `Ideogram4PromptBuilderKJ`.

## Privacy Mode

`AIO Image Generate` and `Ideogram 4 Prompt Builder` include a `privacy_mode` toggle. When enabled, prompt text and prompt-builder editor state are saved to workflow JSON as AES-256-GCM envelopes under the `helto.aio-image-generate` schema, using a local key file at `config/privacy_key.json`. The frontend also masks private text while the node is not hovered and reveals it while the pointer is inside the node.

Encrypted workflows require the same local privacy key to decrypt. If the key is missing or different, the node keeps a locked/error state instead of restoring private text as clear text.

All settings nodes expose `attention_mode` (`auto`, `off`, `sage`, `sage3`, `flash`, `xformers`, `pytorch`, `split`, `sub_quad`), `torch_compile_mode` (`auto`, `off`, `on`), `torch_compile_backend` (`inductor`, `cudagraphs`), and `performance_apply_timing` (`after_loras`, `before_loras`). `auto` attention selects the best installed compatible backend, `off` leaves ComfyUI defaults untouched, and `after_loras` applies attention/compile patches to the final LoRA-patched model.

Krea 2 additionally exposes `fp16_accumulation_enabled`, matching the provided workflow's torch matmul setting behavior when the runtime supports ComfyUI model callbacks.

The main node rejects mismatched settings, for example connecting FLUX settings while `model_type` is `z_image_turbo`.

## LoRA Configuration

`AIO LoRA Configuration` returns an `AIO_LORA_CONFIG` dict for the main node. Its UI is modeled after rgthree's Power LoRA Loader: add ordered LoRA rows, toggle rows, toggle all from the node menu, reorder or remove rows from the row context menu, and choose single or separate model/clip strengths.

LoRAs are applied after the diffusion model and text encoder are loaded, and before prompt encoding and sampling. This matches how a normal workflow would place LoRA loaders after model loading. The backend uses ComfyUI's `LoraLoader.load_lora`, so LoRA files stay lazy and are not loaded at import time.

Ideogram 4 applies LoRAs to the conditional diffusion model only, matching ComfyUI's `LoraLoaderModelOnly` workflow pattern. The Qwen3-VL text encoder and unconditional model are not LoRA-patched by the Ideogram 4 adapter. When `run_unconditional_model` is disabled, the unconditional model is not required, loaded, or patched.

The LoRA info button is also implemented locally. It reads safetensors metadata, stores editable notes/strength hints in a sidecar `*.aio-lora-info.json` file beside the LoRA, can fetch Civitai model-version data by SHA256 hash, and renders a rgthree-style info dialog without requiring rgthree as a runtime dependency.

### Attribution

The LoRA configuration node and LoRA info dialog are inspired by and partially adapted from [rgthree-comfy](https://github.com/rgthree/rgthree-comfy), especially its Power LoRA Loader UI and model-info dialog. rgthree-comfy is copyright Regis Gaughan, III (rgthree) and is distributed under the MIT License.

The Ideogram 4 prompt builder backend formatting and editor behavior are adapted from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)' `Ideogram4PromptBuilderKJ`, which is distributed under GPL-3.0. See `THIRD_PARTY_NOTICES.md`.

The Krea 2 conditioning rebalance helper is adapted from [ComfyUI-ConditioningKrea2Rebalance](https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance), which is distributed under Apache-2.0. See `THIRD_PARTY_NOTICES.md`.

API workflows can pass rgthree-style dynamic row payloads directly:

```json
{
  "show_strengths": "separate",
  "match": "style",
  "lora_1": {
    "on": true,
    "lora": "my_style.safetensors",
    "strength": 0.8,
    "strengthTwo": 0.6
  }
}
```

`run_info.loras` records the enabled, non-zero LoRAs that were applied.

`run_info.performance` records the requested and resolved attention mode, Torch/Triton compile mode, compile backend, and whether performance patches were applied before or after LoRAs.

## Second Sampler Pass

`AIO Image Generate` can run an optional second img2img pass after the normal first-pass image is decoded. When enabled, the node upscales the first-pass image, VAE-encodes the upscaled image, samples it again with the same post-LoRA model, VAE, seed, sampler, scheduler, and conditioning, then decodes the refined output as the main `image`.

Controls:

- `second_pass_enabled`: enable the upscaled refinement pass.
- `second_pass_steps`: second-pass step count. The default `0` reuses the main resolved step count.
- `second_pass_denoise`: second-pass denoise strength, default `0.15`.
- `second_pass_upscale_ratio`: image scale factor, default `1.5`.
- `second_pass_upscale_method`: resize filter, default `lanczos`.

When the pass is enabled, `image_original` exposes the first-pass image before upscaling/refinement. `run_info.second_pass` records whether the pass ran, the second-pass step input and effective step count, the denoise/upscale settings, the first-pass size, and the final refined size.

## External Model Patching

`AIO Load Pipeline Models` loads the same diffusion model and text encoder pair as the main node, applies an optional `AIO_LORA_CONFIG`, then outputs standard ComfyUI `MODEL` and `CLIP` values. Connect those outputs through any compatible external model/CLIP patch nodes, then connect the patched results to the optional `model` and `clip` inputs on `AIO Image Generate`.

When both `model` and `clip` are connected, the main node treats them as already post-LoRA and skips internal model/CLIP loading and LoRA application. This supports external model modifications after AIO LoRAs while keeping the one-node generation path available for simple workflows. If your patch node only modifies `MODEL`, route the loader's `CLIP` output directly into the main node's `clip` input.

## Known Limitations

- GGUF depends on an external compatible backend such as ComfyUI-GGUF.
- Output size is controlled globally with `size mode`, `max side`, `aspect ratio`, and `multiple value`, except when an Ideogram 4 Prompt Builder is connected through Ideogram 4 Settings; in that case the builder dimensions override the main node for Ideogram 4.
- FLUX.2 Klein supports up to four reference images through `image 1` to `image 4`.
- FLUX.2 Klein settings expose reference image scaling controls, defaulting to 1.0 megapixel, `area`, and 1 resolution step. For Flux inpaint on 16 GB GPUs, keeping connected references at or below 1.0 megapixel is recommended.
- FLUX.2 Klein supports inpaint through the dedicated `AIO Inpaint` config node. With `ComfyUI-Inpaint-CropAndStitch` installed, the Flux path uses the shared crop/stitch controls, crops around the mask, uses ComfyUI `InpaintModelConditioning`, samples the working crop, and stitches the decoded result back to the original image size. New `AIO Inpaint` nodes default to crop/stitch-style values: mask grow 8% of the active mask bounding box, mask feather 24px, and a 1024x1024 working crop target. The advanced `source_latent_mode` control can be set to `full image` to bypass crop/stitch and encode the whole source frame as the input latent; the full-frame path still obeys `max_full_frame_megapixels` and `max_full_frame_side` before VAE/sampling. The generation path keeps the crop/stitch node's default GPU mode for parity with the original workflow. The node's `final_mask` output shows the stitcher blend mask projected back to source image size, or the source-size grown and feathered mask on the fallback/full-image path. Crop/stitch mask preview preparation uses CPU mode to avoid occupying Flux sampler VRAM for large source images.
- `AIO Image Generate` exposes optional inpaint debug outputs for the prepared working image, decoded pre-stitch/pre-blend sample, and working mask so crop/stitch behavior can be inspected before final compositing.
- Without the optional crop/stitch node pack, Flux falls back to full-frame masked sampling and final blend. The fallback path downsizes large full-frame inputs using `max_full_frame_megapixels` and `max_full_frame_side` from `AIO Inpaint` before VAE/sampling, so the fallback output may be smaller than the original source.
- FLUX.2 Klein latent-only inpaint output returns the sampled working latent; decoded image output is the path that restores the original canvas size.
- The legacy `mask` input is still only accepted alongside `image 1` and is not the inpaint contract.
- Ideogram 4 supports text-to-image and `AIO Inpaint` in this adapter. With `ComfyUI-Inpaint-CropAndStitch` installed, Ideogram inpaint uses the shared crop/stitch controls by default, samples a clean-source latent with `noise_mask`, and stitches the decoded crop back to the original image size; `source_latent_mode=full image` instead encodes the whole bounded source frame and blends the decoded result directly. It does not use Flux `InpaintModelConditioning`. Reference images, legacy masks, negative prompts, and GGUF model files are not implemented for Ideogram 4.
- Ideogram 4 output dimensions must be multiples of 16, between 256 and 2048 pixels per side, with aspect ratio no wider than 6:1.
- Krea 2 supports text-to-image and `AIO Inpaint` in this adapter. The Krea inpaint path uses the shared crop/stitch controls by default, samples a clean-source latent with `noise_mask`, and stitches or blends the decoded result back into the source image; `source_latent_mode=full image` instead encodes the whole bounded source frame and blends the decoded result directly. It does not use Flux `InpaintModelConditioning`. Reference images, legacy masks, and negative prompts are not implemented for Krea 2. GGUF requires a compatible external backend and Krea-compatible GGUF model files.
- Krea 2 output dimensions must be multiples of 16.
- Z-Image reference-image and mask paths are staged for a later adapter pass.

## Adapter Implementation Notes

Local ComfyUI source was inspected before implementing the generation pipeline. Relevant references:

- `/home/thhel/git/ComfyUI/nodes.py`: classic `NODE_CLASS_MAPPINGS`, `folder_paths`, `UNETLoader`, `CLIPLoader`, `VAEEncode`, `VAEDecode`, `ImageScaleBy`, `common_ksampler`, `InpaintModelConditioning`
- `/home/thhel/git/ComfyUI/comfy/sample.py`: `comfy.sample.sample` signature
- `/home/thhel/git/ComfyUI/comfy/samplers.py`: sampler `noise_mask` and denoise-mask behavior
- `/home/thhel/git/ComfyUI/comfy/utils.py`: `ProgressBar`, `common_upscale`
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_flux.py`: `EmptyFlux2LatentImage`, Flux guidance, Flux2 scheduler
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_ideogram4.py`: Ideogram 4 scheduler and sigma helper
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_custom_sampler.py`: `DualModelGuider`, `CFGOverride`, `RandomNoise`, `KSamplerSelect`, `SamplerCustomAdvanced`, `BasicScheduler`
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_model_advanced.py`: `ModelSamplingAuraFlow`
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_zimage.py`: Z-Image conditioning node patterns
- `/home/thhel/git/ComfyUI/comfy/text_encoders/krea2.py`: Krea 2 flattened text-conditioning shape
- `/home/thhel/git/ComfyUI/comfy/model_base.py`: Krea 2 extra conditioning fields and Flux/Flux2 concat inpaint conditioning
- `/home/thhel/git/ComfyUI/comfy/supported_models.py`: `Flux2`, `Ideogram4`, `Krea2`, and `ZImage` model-family detection
- `/home/thhel/git/ComfyUI/nodes.py`: `LoraLoader.load_lora`, `LoraLoaderModelOnly`, `CLIPTextEncode`
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/py/power_lora_loader.py`: Power LoRA dynamic backend payload shape
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/comfyui/power_lora_loader.js`: Power LoRA frontend interaction model
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/comfyui/dialog_info.js`: rgthree LoRA info dialog behavior
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/common/css/dialog_model_info.css`: rgthree LoRA info dialog layout and visual styling
- `/home/thhel/git/ComfyUI/custom_nodes/comfyui-kjnodes/nodes/ideogram4_nodes.py`: Ideogram 4 structured prompt JSON shape and preview/bbox behavior
- `/home/thhel/git/ComfyUI/custom_nodes/comfyui-kjnodes/web/js/ideogram4_prompt_builder.js`: Ideogram 4 prompt builder frontend interaction model
- `/home/thhel/git/ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py`: `UnetLoaderGGUF`, `CLIPLoaderGGUF`, `clip_gguf`, and `unet_gguf` path/list patterns
- `/home/thhel/git/ComfyUI/custom_nodes/gguf/pig.py`: `clip_gguf`, `model_gguf`, and `vae_gguf` path/list patterns
- `/home/thhel/git/ComfyUI/custom_nodes/comfyui-inpaint-cropandstitch/inpaint_cropandstitch.py`: optional `InpaintCropImproved` / `InpaintStitchImproved` crop-stitch node contract

The adapters compose these existing primitives lazily at execution time so importing the node pack does not load ComfyUI models, torch, or GGUF packages.

## How To Add A New Model Family

1. Add a profile in `services/profiles.py`.
2. Add an adapter in `adapters/` and decorate it with `register_adapter`.
3. Optionally add a settings node that returns `AIO_MODEL_SETTINGS`.
4. Register the settings node in root `__init__.py`.
5. Add tests for defaults, validation, settings, and registry behavior.

## Nodes V3 Migration Notes

This pack uses classic nodes for broad community compatibility. The contracts are intentionally kept clean for a future V3 wrapper: no frontend-only execution state, no hidden mutable globals, and model-family logic isolated behind adapters and services.

## License Guidance

The original AIO code is distributed under the MIT License. Because the LoRA UI intentionally adapts behavior and styling from rgthree-comfy, MIT matches rgthree-comfy's license, keeps the same permissive terms, and preserves the original rgthree MIT copyright and permission notice in `THIRD_PARTY_NOTICES.md`.

The Ideogram 4 prompt-builder implementation includes code and behavior adapted from GPL-3.0 KJNodes sources. Treat those derived prompt-builder portions as GPL-3.0-covered material, comply with GPL-3.0 when redistributing them, and keep the KJNodes notice intact.

The Krea 2 conditioning rebalance implementation includes code and behavior adapted from Apache-2.0 ComfyUI-ConditioningKrea2Rebalance sources. Preserve the Apache-2.0 attribution and notice when redistributing those derived rebalance portions.

For new code that does not derive from GPL sources, MIT remains the simplest and most community-friendly fit for this ComfyUI node pack.
