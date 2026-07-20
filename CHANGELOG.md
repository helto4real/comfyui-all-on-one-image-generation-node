# Changelog

## Unreleased

- Prevented private prompt text from leaking through run-info debug data, execution-history UI payloads, or external ComfyUI caches.
- Confined LoRA metadata, sidecar, and preview access to configured LoRA roots; made Civitai access an explicit refresh action; moved blocking metadata work off the async server loop; and escaped all untrusted dialog content.
- Corrected FLUX.2 base defaults to 50 steps while retaining 4-step distilled defaults and explicit step overrides.
- Removed settings controls that did not affect execution while migrating old serialized widget arrays to the remaining controls.
- Restored ComfyUI's original reserved-VRAM value after leaving low-VRAM policies.
- Kept fresh package imports independent of Torch, GGUF, and ComfyUI server imports.
- Corrected dependency installation guidance and aligned the combined distribution license with its GPL-3.0-derived prompt-builder code.

## 0.1.0

- Added classic ComfyUI node registration for AIO Image Generate.
- Added Z-Image Turbo and FLUX.2 Klein 9B settings nodes.
- Added model profiles, adapter registry, validation, progress, run-info, and loader backend scaffolds.
- Added real text-to-image generation paths for Z-Image Turbo and FLUX.2 Klein 9B using local ComfyUI loader, encoder, sampler, and VAE decode primitives.
- Added FLUX.2 Klein support for `AIO Inpaint` masked img2img with optional reference images, Flux inpaint conditioning, optional crop/stitch sampling, and original-canvas restoration for decoded image output.
- Tuned shared `AIO Inpaint` crop/stitch controls toward the Flux crop/stitch workflow, including percent-based mask grow, 24px blend, and a 1024x1024 working crop target for new nodes.
- Added an `AIO Inpaint` `final_mask` output that exposes the source-size blend mask after grow and feathering.
- Added `AIO Image Generate` inpaint debug outputs for the working source image, decoded pre-blend sample, and working mask.
- Made Flux inpaint safer for large sources by using CPU crop/stitch preparation for the `final_mask` preview, capping full-frame fallback inputs before sampling, and keeping crop/stitch generation on the original GPU path for workflow parity.
- Added Ideogram 4 `AIO Inpaint` crop/stitch support using clean-source latent sampling with `noise_mask`.
- Added runtime GGUF loader-node integration for diffusion and text encoder loading through compatible GGUF custom nodes.
- Added AIO LoRA Configuration node with dynamic ordered LoRA rows and runtime application through ComfyUI's LoRA loader.
- Added optional upscaled second sampler pass with configurable steps, `image_original` output, and `run_info.second_pass` metadata.
- Grouped internal `AIO Image Generate` outputs into `model_info`, `pid_info`, and `inpaint_info`, with new `AIO Model Info`, `AIO PID Info`, and `AIO Inpaint Info` utility nodes for unpacking the old direct values.
- Added tests that do not require real model files.
