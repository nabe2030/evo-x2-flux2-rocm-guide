# Running FLUX.2 dev on EVO-X2 (Ryzen AI Max+ 395 / gfx1151) with ROCm — Practical Guide

> **This guide reflects the state of things as of late March 2026.** Kernel updates, ROCm releases, and ComfyUI changes may alter what works.

## Who This Is For

You have an EVO-X2 (or another Strix Halo device with Ryzen AI Max+ 395), and you want to run FLUX.2 dev image generation locally. You've probably already discovered that:

- ROCm doesn't officially support gfx1151 (RDNA 3.5)
- Standard FP8 safetensors models crash immediately
- BF16 full-precision models don't fit in memory
- Most guides written for MI300X or RDNA 3 GPUs don't apply

This guide documents a **working configuration** verified through extended stability testing (6+ consecutive generations, zero crashes), and catalogs the configurations that **don't work** so you can skip repeating these experiments.

> **Scope:** This article covers basic text-to-image generation only. LoRA application will be covered in Part 2.

## Environment

### Hardware

| Component | Spec |
|---|---|
| Machine | GMKtec EVO-X2 |
| CPU/GPU | AMD Ryzen AI Max+ 395 with Radeon 8060S |
| GPU arch | gfx1151 (RDNA 3.5) |
| Total RAM | 128 GB |
| BIOS VGM setting | 64 GB (→ VRAM 64 GB / CPU RAM ~62 GB) |
| GPU memory bandwidth | ~218 GB/s (intra-GPU) |

**VGM = 64 GB was the only stable setting in our testing.** VGM=96GB reduces CPU RAM to ~30GB and causes OOM kills during text encoder loading. VGM=512MB with GTT kernel parameters caused model placement in slow memory and hangs.

### Software

| Component | Version |
|---|---|
| OS | Ubuntu 25.10 (Questing Quokka) Server |
| Kernel | 6.18.20-061820-generic (Mainline PPA) |
| Python | 3.13.7 |
| ROCm | 7.13.0a (TheRock nightly, pip wheel) |
| PyTorch | 2.10.0+rocm7.13.0a |
| Triton | 3.6.0+rocm7.13.0a |
| ComfyUI | Latest (with ComfyUI-GGUF custom node) |

**Why Ubuntu 25.10 + kernel 6.18?** Kernel 6.14 (Ubuntu 24.04 OEM) had SDMA/SVM-related crashes in multi-model workflows. Kernel 6.18 resolves these, eliminating the need for `HSA_USE_SVM=0`. Basic single-model generation works on either kernel, but 6.18 provides a significantly more stable foundation.

**Why TheRock nightly pip wheels?** gfx1151 is not included in official ROCm releases. TheRock nightly builds include gfx1151 support and can be installed via pip without a system-level ROCm installation.

### Setup Commands

```bash
# Create Python 3.13 venv
python3.13 -m venv ~/rocm-therock-25
source ~/rocm-therock-25/bin/activate

# Install TheRock nightly wheels (gfx1151 support)
# NOTE: URL may change. Check https://github.com/ROCm/TheRock for latest instructions.
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1

# Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI && pip install -r requirements.txt

# Install ComfyUI-GGUF (required for GGUF model loading)
cd ~/ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF && pip install -r requirements.txt
```

### Known Constraints of gfx1151 (as of late March 2026)

These are **hardware or driver limitations** that no software configuration could work around in our testing:

1. **No FP8 compute** — `torch._scaled_mm` requires CDNA3 (MI300+). Not supported on gfx1151.
2. **mmap unstable above 64 GB** — A ROCm bug causes mmap of files larger than ~64 GB to hang or become extremely slow.
3. **AOTriton Flash Attention has no effect** — Setting `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables `flash_sdp`, but produced zero speed improvement for FLUX.2 inference in our testing. It can also trigger `MEMORY_APERTURE_VIOLATION` crashes.
4. **Q5_K dequantization crashes** — The bit-shift operations in Q5_K GGUF decoding are unstable on gfx1151. Q8_0 is stable.

## Working Configuration

### Model Files

| Role | File | Size | Directory |
|---|---|---|---|
| Diffusion model | `flux2-dev-Q8_0.gguf` | 35 GB | `models/diffusion_models/` |
| Text encoder | `mistral_3_small_flux2_fp8.safetensors` | 17 GB | `models/text_encoders/` |
| VAE | `flux2-vae.safetensors` | 321 MB | `models/vae/` |

Paths are relative to `~/ComfyUI/`.

> ℹ️ FLUX.2 dev uses a Mistral-based text encoder (17 GB), not the T5XXL + CLIP_L combination from FLUX.1. Set the CLIPLoader node type to `flux2`.

> ⚠️ Use `flux2-vae.safetensors` (FLUX.2), not `ae.safetensors` (FLUX.1). The FLUX.1 VAE produces a 16ch/128ch channel mismatch error.

### Launch Script

```bash
#!/bin/bash
# FLUX.2 dev on EVO-X2 (gfx1151) — verified stable, 2026-03-29

source ~/rocm-therock-25/bin/activate

export HIP_VISIBLE_DEVICES=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

cd ~/ComfyUI
python main.py \
    --listen 0.0.0.0 \
    --port 8190 \
    --disable-mmap \
    --bf16-vae \
    --cache-none
```

### What Each Option Does

| Option | Purpose | What happens without it |
|---|---|---|
| `--disable-mmap` | **Required on Strix Halo.** Avoids the ROCm mmap bug with files >64 GB | Model loading hangs or becomes extremely slow |
| `--bf16-vae` | Runs the VAE in BF16 precision | — |
| `--cache-none` | Disables model caching in VRAM | Out-of-memory crashes in multi-model workflows |
| `HIP_VISIBLE_DEVICES=0` | Selects the GPU device | — |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | Enables AOTriton (experimental) | No measurable effect on FLUX.2, but may benefit other workloads |

### Performance

| Metric | Value |
|---|---|
| Speed | 27.16 s/step |
| Total time (1024×1024, 28 steps) | ~12 min 40 s |
| Stability | 6+ consecutive generations, zero crashes |

#### Measurement Conditions

| Parameter | Value |
|---|---|
| Resolution | 1024×1024 |
| Steps | 28 |
| Sampler | euler |
| Scheduler | simple |
| CFG | 1.0 (FLUX.2 dev recommended) |
| Seed | Fixed (for reproducibility) |
| Workflow | ComfyUI txt2img (Unet Loader GGUF → CLIPLoader type:flux2 → KSampler → VAE Decode) |
| Method | Step times from ComfyUI console output, averaged over 28 steps |

## Configurations That Don't Work

As of late March 2026, these have all been tested and failed in our environment:

| Model | Size | VGM | Options | Result | Root Cause |
|---|---|---|---|---|---|
| FP8 safetensors (dtype=fp8) | 34 GB | 96 GB | — | Segfault | `_scaled_mm` requires MI300+ |
| FP8 safetensors (dtype=default) | 34 GB | 96 GB | — | Segfault | Internal FP8 compute in MixedPrecisionOps |
| BF16 safetensors | 61 GB | 96 GB | `--disable-smart-memory` | OOM Killed | Total model footprint exceeds 96 GB |
| BF16 safetensors | 61 GB | 96 GB | smart memory | Segfault | FP8 metadata issue |
| BF16 safetensors | 61 GB | 96 GB | — | mmap failure | Cannot mmap 61 GB with 32 GB CPU RAM |
| BF16 safetensors | 61 GB | 64 GB | `--disable-mmap` | Killed | 61 GB model + OS doesn't fit in 62 GB RAM |
| Q5_K_M GGUF | 24 GB | 96 GB | — | Crash at step 9 | Q5_K dequant bit-shift bug on gfx1151 |
| Q8_0 GGUF | 35 GB | 96 GB | — | Crash at step 3–5 | Only 32 GB CPU RAM at VGM=96GB |
| AOTriton Flash Attention | Q8_0 | 64 GB | `AOTRITON_EXPERIMENTAL=1` | Crash | `MEMORY_APERTURE_VIOLATION` |
| GTT kernel params | Q8_0 | 64 GB | `amdgpu.gttsize=126976` | Hang | Model placed in GTT (slow path) |

In our testing, **Q8_0 GGUF with VGM=64GB was the only viable path.** FP8 lacks hardware support, BF16 doesn't fit in memory, and Q5_K has unstable dequantization. Future ROCm or kernel updates may change this.

## Performance Reality

### Reference Comparison with DGX Spark

For context, here's a comparison with NVIDIA DGX Spark (GB10 Blackwell). **Note: this is not an apples-to-apples comparison** — DGX Spark uses FP8/BF16 safetensors while EVO-X2 uses Q8_0 GGUF. These numbers represent "best available configuration on each platform" as a practical reference.

| | DGX Spark (FP8) | DGX Spark (BF16) | EVO-X2 (Q8_0 GGUF) |
|---|---|---|---|
| Step speed | 5.39 s/step | 8.34 s/step | 27.16 s/step |
| Total (1024×1024, 28 steps) | 153 s | 434 s | 762 s |
| Relative speed | 1.0× | 0.65× | 0.20× |

### Why This Appears to Be Near the Ceiling (as of late March 2026)

The performance gap is fundamentally hardware-bound:

- gfx1151 (RDNA 3.5) delivers ~46 TFLOPS BF16 effective ([The Register benchmark](https://www.theregister.com/))
- DGX Spark (Blackwell GB10) delivers 120–125 TFLOPS — roughly 2.5× faster
- 27 s/step with Q8_0 GGUF + math attention appears to be near the theoretical maximum for this hardware

**About AMD's "5× faster" claim:** At CES 2026, AMD announced ROCm 7.1 is "5× faster than ROCm 6.4." This comparison is against ROCm 6.4.4, which had unoptimized gfx1151 kernels. It does not apply to current environments running ROCm 7.13.

## Tuning Attempts

### Helped

| Setting | Effect |
|---|---|
| Ubuntu 25.10 + kernel 6.18.20 | Resolved crashes in multi-model workflows (`HSA_USE_SVM=0` no longer needed) |
| ROCm 7.13 TheRock nightly (cp313) | Performance parity with 7.12, adds Python 3.13 support |
| `--cache-none` | Required to prevent out-of-memory in multi-model workflows |

### No Effect (zero speed change in our testing)

| Setting | Result |
|---|---|
| ROCm 7.12 → 7.13 upgrade | No speed change |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | No speed change for FLUX.2 inference |
| `--use-pytorch-cross-attention` | No speed change |
| `ROCBLAS_USE_HIPBLASLT=1` | No speed change |
| `MIOPEN_FIND_MODE=FAST` | No speed change |

### Made Things Worse

| Setting | Result |
|---|---|
| GRUB: `amdgpu.gttsize=126976 ttm.pages_limit=32505856` | Model placed in GTT (slow memory), generation hangs. **Removed.** |
| VGM=96GB (BIOS) | CPU RAM drops to ~30 GB → OOM Kill. **Reverted to 64 GB.** |
| Flash Attention via AOTriton | `MEMORY_APERTURE_VIOLATION` crash |
| Removing `--cache-none` | Out-of-memory crash in multi-model workflows |

## Recommended Setup (TL;DR)

As of late March 2026:

1. Set VGM to 64 GB in BIOS
2. Install Ubuntu 25.10 with kernel 6.18+
3. Create a Python 3.13 venv and install TheRock nightly ROCm pip wheels for gfx1151
4. Install ComfyUI + ComfyUI-GGUF custom node
5. Download: `flux2-dev-Q8_0.gguf`, `mistral_3_small_flux2_fp8.safetensors`, `flux2-vae.safetensors`
6. Launch with: `--disable-mmap --bf16-vae --cache-none`

**Expect ~27 s/step at 1024×1024 (28 steps), ~12.5 minutes per image.**

## Related Resources

- [Strix Halo ComfyUI Toolbox](https://github.com/kyuz0/amd-strix-halo-comfyui-toolboxes) — community toolbox with benchmarks
- [AMD Strix Halo AI Guide](https://github.com/bkpaine1/AMD-Strix-Halo-AI-Guide) — comprehensive setup guide
- [Docker-based setup (rocm/pytorch)](https://github.com/IgnatBeresnev/comfyui-gfx1151)
- [Strix Halo Wiki](https://strixhalo.wiki/)
- [AMD x ComfyUI blog](https://www.amd.com/en/blogs/2026/amd-comfyui-advancing-professional-quality-generative-ai-ryzen-radeon.html)
- [ROCm Strix Halo optimization docs](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)

## Next: Part 2 — LoRA Application

Part 2 will cover applying custom LoRA models trained on DGX Spark (CUDA) to EVO-X2 (ROCm) inference, including performance impact and workflow configuration.

---

*Tested on: GMKtec EVO-X2, Ryzen AI Max+ 395, Ubuntu 25.10, kernel 6.18.20, ROCm 7.13 TheRock nightly, 2026-03-29*
