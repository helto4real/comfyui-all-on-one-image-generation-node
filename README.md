# AIO Image Generate

A classic ComfyUI custom node pack for an extensible all-in-one image generation node. The visible node is a simple facade, while model-family behavior lives in profiles, adapters, loader backends, validation services, and shared progress utilities.

## Installation

Clone or copy this folder into `ComfyUI/custom_nodes`, then restart ComfyUI. The pack has no extra runtime dependency beyond ComfyUI itself.

## Nodes

- `AIO Image Generate`
- `Z-Image Turbo Settings`
- `FLUX.2 Klein 9B Settings`
- `AIO LoRA Configuration`

All nodes appear under `AIO/Image`.

## Supported Model Families

- `z_image_turbo`: text-to-image generation, defaults to 8 steps and CFG 1.0. Negative prompts are ignored by default and reported in `run_info.warnings`.
- `flux2_klein_9b`: text-to-image generation, distilled defaults to 4 steps and CFG 1.0. Reference editing settings are accepted by the settings node but are not implemented yet.

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

The dropdown may prefix values with their category when multiple folders are searched.

## Basic Usage

1. Add `AIO Image Generate`.
2. Select `model_type`.
3. Select diffusion model, text encoder, and VAE files.
4. Enter a positive prompt.
5. Optionally attach the matching model-specific settings node.
6. Optionally attach `AIO LoRA Configuration` to apply one or more LoRAs.
7. Connect `IMAGE` to Preview Image or Save Image.

The node is not an output node, so it is safe for API-mode workflows.

## Settings Nodes

`Z-Image Turbo Settings` returns an `AIO_MODEL_SETTINGS` dict with speed preset, forced steps, prompt enhancement, negative-prompt policy, and precision policy.

`FLUX.2 Klein 9B Settings` returns an `AIO_MODEL_SETTINGS` dict with distilled/base variant, guidance, edit mode, reference strength, precision policy, memory policy, and shift parameters.

The main node rejects mismatched settings, for example connecting FLUX settings while `model_type` is `z_image_turbo`.

## LoRA Configuration

`AIO LoRA Configuration` returns an `AIO_LORA_CONFIG` dict for the main node. Its UI is modeled after rgthree's Power LoRA Loader: add ordered LoRA rows, toggle rows, toggle all from the node menu, reorder or remove rows from the row context menu, and choose single or separate model/clip strengths.

LoRAs are applied after the diffusion model and text encoder are loaded, and before prompt encoding and sampling. This matches how a normal workflow would place LoRA loaders after model loading. The backend uses ComfyUI's `LoraLoader.load_lora`, so LoRA files stay lazy and are not loaded at import time.

The LoRA info button is also implemented locally. It reads safetensors metadata, stores editable notes/strength hints in a sidecar `*.aio-lora-info.json` file beside the LoRA, can fetch Civitai model-version data by SHA256 hash, and renders a rgthree-style info dialog without requiring rgthree as a runtime dependency.

### Attribution

The LoRA configuration node and LoRA info dialog are inspired by and partially adapted from [rgthree-comfy](https://github.com/rgthree/rgthree-comfy), especially its Power LoRA Loader UI and model-info dialog. rgthree-comfy is copyright Regis Gaughan, III (rgthree) and is distributed under the MIT License.

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

## Known Limitations

- GGUF depends on an external compatible backend such as ComfyUI-GGUF.
- Output size is controlled globally with `size mode`, `max side`, and `aspect ratio`.
- FLUX.2 Klein supports up to four reference images through `image 1` to `image 4`.
- FLUX.2 Klein accepts a mask with `image 1`, but inpaint behavior is staged for a later adapter pass.
- Z-Image reference-image and mask paths are staged for a later adapter pass.

## Adapter Implementation Notes

Local ComfyUI source was inspected before implementing the generation pipeline. Relevant references:

- `/home/thhel/git/ComfyUI/nodes.py`: classic `NODE_CLASS_MAPPINGS`, `folder_paths`, `UNETLoader`, `CLIPLoader`, `VAEDecode`, `common_ksampler`
- `/home/thhel/git/ComfyUI/comfy/sample.py`: `comfy.sample.sample` signature
- `/home/thhel/git/ComfyUI/comfy/utils.py`: `ProgressBar`
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_flux.py`: `EmptyFlux2LatentImage`, Flux guidance, Flux2 scheduler
- `/home/thhel/git/ComfyUI/comfy_extras/nodes_zimage.py`: Z-Image conditioning node patterns
- `/home/thhel/git/ComfyUI/comfy/supported_models.py`: `Flux2` and `ZImage` model-family detection
- `/home/thhel/git/ComfyUI/nodes.py`: `LoraLoader.load_lora`
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/py/power_lora_loader.py`: Power LoRA dynamic backend payload shape
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/comfyui/power_lora_loader.js`: Power LoRA frontend interaction model
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/comfyui/dialog_info.js`: rgthree LoRA info dialog behavior
- `/home/thhel/git/ComfyUI/custom_nodes/rgthree-comfy/web/common/css/dialog_model_info.css`: rgthree LoRA info dialog layout and visual styling
- `/home/thhel/git/ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py`: `UnetLoaderGGUF`, `CLIPLoaderGGUF`, `clip_gguf`, and `unet_gguf` path/list patterns
- `/home/thhel/git/ComfyUI/custom_nodes/gguf/pig.py`: `clip_gguf`, `model_gguf`, and `vae_gguf` path/list patterns

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

This project is distributed under the MIT License. Because the LoRA UI intentionally adapts behavior and styling from rgthree-comfy, MIT matches rgthree-comfy's license, keeps the same permissive terms, and preserves the original rgthree MIT copyright and permission notice in `THIRD_PARTY_NOTICES.md`.

Other compatible options include Apache-2.0, if you want an explicit patent grant, or GPL-3.0, if you want downstream copyleft requirements. For this ComfyUI node pack, MIT is the simplest and most community-friendly fit.
