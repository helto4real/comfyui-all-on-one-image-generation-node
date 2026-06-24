# Changelog

## 0.1.0

- Added classic ComfyUI node registration for AIO Image Generate.
- Added Z-Image Turbo and FLUX.2 Klein 9B settings nodes.
- Added model profiles, adapter registry, validation, progress, run-info, and loader backend scaffolds.
- Added real text-to-image generation paths for Z-Image Turbo and FLUX.2 Klein 9B using local ComfyUI loader, encoder, sampler, and VAE decode primitives.
- Added FLUX.2 Klein support for `AIO Inpaint` masked img2img with optional reference images, Flux inpaint conditioning, optional crop/stitch sampling, and original-canvas restoration for decoded image output.
- Tuned shared `AIO Inpaint` crop/stitch controls toward the Flux crop/stitch workflow, including 16px mask grow, 24px blend, and a 1024x1024 working crop target for new nodes.
- Added Ideogram 4 `AIO Inpaint` crop/stitch support using clean-source latent sampling with `noise_mask`.
- Added runtime GGUF loader-node integration for diffusion and text encoder loading through compatible GGUF custom nodes.
- Added AIO LoRA Configuration node with dynamic ordered LoRA rows and runtime application through ComfyUI's LoRA loader.
- Added tests that do not require real model files.
