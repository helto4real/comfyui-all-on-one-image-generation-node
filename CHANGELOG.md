# Changelog

## 0.1.0

- Added classic ComfyUI node registration for AIO Image Generate.
- Added Z-Image Turbo and FLUX.2 Klein 9B settings nodes.
- Added model profiles, adapter registry, validation, progress, run-info, and loader backend scaffolds.
- Added real text-to-image generation paths for Z-Image Turbo and FLUX.2 Klein 9B using local ComfyUI loader, encoder, sampler, and VAE decode primitives.
- Added runtime GGUF loader-node integration for diffusion and text encoder loading through compatible GGUF custom nodes.
- Added AIO LoRA Configuration node with dynamic ordered LoRA rows and runtime application through ComfyUI's LoRA loader.
- Added tests that do not require real model files.
