---
name: diffusion-models
description: >
  How to add and work with diffusion / image-generation models in
  mobius. Covers UNet, VAE, DiT, Flux, SD3, ControlNet,
  adapters, and QwenImage architectures. Includes pipeline detection,
  diffusers configs, task classes, building blocks, and weight loading
  conventions. Use this skill when adding or modifying a diffusion model.
---

# Skill: Diffusion Models

## When to use

Use this skill when:
- Adding a new diffusion model (denoiser, VAE, ControlNet, adapter)
- Working with `build_diffusers_pipeline()` or the `_DIFFUSERS_CLASS_MAP`
- Creating diffusers config classes or task types
- Debugging diffusers weight loading or pipeline detection

## Architecture overview

Diffusion models generate images/video by iteratively denoising latent
representations. The key components are:

| Component | Role | Examples |
|-----------|------|----------|
| **Denoiser** | Predicts noise to remove at each step | UNet2D, DiT, Flux, SD3, QwenImage |
| **VAE** | Encodes images to latent / decodes latent to images | AutoencoderKL, QwenImage 3D VAE, Video VAE |
| **ControlNet** | Provides spatial conditioning (edges, depth, etc.) | ControlNetModel |
| **Adapter** | Adds image/conditioning features to denoiser | T2I-Adapter, IP-Adapter |

## Pipeline detection and building

### How it works

When the CLI or `build()` encounters a model that isn't a transformers model,
it checks for `model_index.json` (the diffusers pipeline descriptor):

```python
# In _diffusers_builder.py
# 1. Try transformers AutoConfig → if fails:
# 2. Try loading model_index.json → if found:
# 3. Parse components and build each via _DIFFUSERS_CLASS_MAP
```

### Registering a new diffusers model

Add an entry to `_init_diffusers_class_map()` in `_diffusers_builder.py`:

```python
def _init_diffusers_class_map():
    _DIFFUSERS_CLASS_MAP["MyTransformer2DModel"] = (
        MyTransformer2DModel,  # Module class
        MyConfig,              # Config dataclass
        "denoising",           # Task name
    )
```

The key must match the class name in `model_index.json`:
```json
{
    "transformer": ["diffusers", "MyTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"]
}
```

### Weight loading for diffusers

Diffusers uses different weight file naming than transformers:

| Convention | Filename |
|------------|----------|
| Single file | `diffusion_pytorch_model.safetensors` |
| Sharded index | `diffusion_pytorch_model.safetensors.index.json` |
| Fallback | `model.safetensors` (some models) |

Weights live in component subdirectories: `transformer/`, `vae/`, etc.
The `_download_diffusers_component_weights()` function tries both naming
conventions.

## Config classes

Diffusers configs are separate from `ArchitectureConfig`. Each is a
`@dataclasses.dataclass` with a `from_diffusers(config: dict)` classmethod
that parses the JSON config.

```python
@dataclasses.dataclass
class MyDiffuserConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    num_layers: int = 2
    attention_head_dim: int = 8

    @classmethod
    def from_diffusers(cls, config: dict) -> "MyDiffuserConfig":
        return cls(
            in_channels=config.get("in_channels", cls.in_channels),
            out_channels=config.get("out_channels", cls.out_channels),
            block_out_channels=tuple(config.get("block_out_channels", cls.block_out_channels)),
            num_layers=config.get("num_layers", cls.num_layers),
            attention_head_dim=config.get("attention_head_dim", cls.attention_head_dim),
        )
```

Existing config classes:
- `VAEConfig` — Standard SD VAE (2D)
- `UNet2DConfig` — UNet-based denoisers
- `DiTConfig` — Diffusion Transformer
- `SD3Config` — Stable Diffusion 3 / MMDiT
- `FluxConfig` — Flux transformer
- `ControlNetConfig` — ControlNet conditioning
- `T2IAdapterConfig` / `IPAdapterConfig` — Adapters
- `QwenImageConfig` — QwenImage transformer
- `QwenImageVAEConfig` — QwenImage 3D causal VAE
- `VideoVAEConfig` — 3D video VAE

All defined in `src/mobius/_diffusers_configs.py`.

## Task classes

Each diffusion model type has a task class that defines I/O signatures:

### DenoisingTask

```python
# Inputs:
#   sample: [B, in_channels, H, W]       — noisy latent
#   timestep: [B]                         — diffusion timestep
#   encoder_hidden_states: [B, seq, dim]  — text conditioning
# Output:
#   noise_pred: [B, in_channels, H, W]   — predicted noise
```

### VAETask

Returns `ModelPackage` with two sub-models:
- `encoder`: `sample [B, 3, H, W]` → `latent_dist [B, 2*latent_ch, H/f, W/f]`
- `decoder`: `latent [B, latent_ch, H/f, W/f]` → `sample [B, 3, H, W]`

### QwenImageVAETask

3D causal VAE for video/images:
- `encoder`: `[B, 3, T, H, W]` → `[B, 2*z_dim, T', H', W']`
- `decoder`: `[B, z_dim, T', H', W']` → `[B, 3, T, H, W]`

### ControlNetTask

```python
# Inputs: sample, timestep, encoder_hidden_states, controlnet_cond
# Outputs: down_outputs (list of residuals), mid_output
```

### AdapterTask

```python
# T2I: condition [B, in_channels, H, W] → feature_list
# IP: image_embeds [B, image_dim] → adapter_output [B, num_tokens, cross_dim]
```

## Building blocks

### Shared components (from `components/`)

Diffusion models import shared primitives from the component library:

| Component | Import | Purpose |
|-----------|--------|---------|
| `Conv2d` | `components/_conv.py` | 2D convolution with bias |
| `GroupNorm` | `components/_common.py` | Group normalization |
| `LayerNormNoAffine` | `components/_common.py` | Norm without learnable params (AdaLN) |
| `Linear` | `components/_common.py` | Linear projection |
| `SiLU` | `components/_activations.py` | SiLU activation module |

Model files import these with underscore-prefixed aliases:

```python
from mobius.components import Conv2d as _Conv2d, GroupNorm as _GroupNorm, SiLU as _SiLU
```

### Common model-specific blocks

| Block | Purpose | Used by |
|-------|---------|---------|
| `_TimestepEmbedding` | Sinusoidal → MLP time embedding | UNet, DiT, Flux, SD3, QwenImage |
| `nn.ModuleList` | Layer lists, skip connections | All |
| `nn.Sequential` | Callable container that chains forward calls | Diffusion `to_out`, `img_mod`, `net` |

> **`nn.Sequential` vs `nn.ModuleList`**: `nn.Sequential` chains children's
> `forward()` calls automatically — prefer it for sequential containers like
> `to_out` and modulation layers. Use `nn.ModuleList` for layer lists that
> need custom iteration (e.g., `down_blocks`, `transformer_blocks`).

### UNet-specific

| Block | Purpose |
|-------|---------|
| `_ResNetBlock2DWithTime` | ResNet block with time embedding injection |
| `_CrossAttentionBlock` | Self-attention + cross-attention + FFN |
| `_BasicAttention` | Scaled dot-product attention (spatial) |
| `_DownBlock2D` | ResNet + optional attn + spatial downsample |
| `_UpBlock2D` | ResNet + optional attn + spatial upsample + skip concat |
| `_UNetMidBlock2DCrossAttn` | Mid block: ResNet → cross-attn → ResNet |

### DiT / Transformer denoiser patterns

| Block | Purpose |
|-------|---------|
| `_PatchEmbed` | Conv2d with stride=patch_size (patchify input) |
| `_AdaLayerNormZero` | Adaptive LayerNorm with scale/shift/gate from timestep |
| `_DiTBlock` | AdaLN-Zero → self-attn + cross-attn + FFN |
| `_JointAttentionBlock` | Concatenate text+image → joint attention → split |
| `_FluxSingleBlock` | Unified self-attention (single-stream) |

### VAE-specific

| Block | Purpose |
|-------|---------|
| `_ResNetBlock2D` | GroupNorm → SiLU → Conv → skip |
| `_AttentionBlock` | Self-attention in latent space |
| `_Downsample2D` / `_Upsample2D` | Spatial resampling |
| `_MidBlock2D` | ResNet + optional attention |

### 3D / Video-specific

| Block | Purpose |
|-------|---------|
| `_CausalConv3d` | 3D conv with temporal-causal padding |
| `_RMSNorm3d` | Channel-wise RMS normalization for 3D |
| `_ResidualBlock` (3D) | 3D ResNet with RMSNorm |
| `_Resample` | 2D spatial + optional 3D temporal resampling |

## Denoiser architecture comparison

| Model | Stream | Attention | Modulation | Normalization |
|-------|--------|-----------|-----------|---------------|
| **UNet2D** | Single | Cross-attn only | Time embed inject | GroupNorm |
| **DiT** | Single | Self + cross | AdaLN-Zero (6 params) | LayerNorm |
| **SD3 (MMDiT)** | Double | Joint (concat text+img) | AdaLN-Zero | LayerNorm |
| **Flux** | Double→Single | Joint then unified | AdaLN-Zero | LayerNorm |
| **QwenImage** | Double | Joint (separate proj) | AdaLN (2×6 params) | RMSNorm |

### Double-stream pattern

SD3, Flux, and QwenImage use a double-stream architecture:
1. **Image stream**: modulation → attention → FFN → residual
2. **Text stream**: separate modulation → attention → FFN → residual
3. **Joint attention**: concatenate Q/K/V from both streams, attend together

```python
# In _JointAttentionBlock.forward():
img_q, img_k, img_v = self.to_q(op, img), self.to_k(op, img), self.to_v(op, img)
txt_q, txt_k, txt_v = self.add_q(op, txt), self.add_k(op, txt), self.add_v(op, txt)
q = op.Concat(img_q, txt_q, axis=1)
k = op.Concat(img_k, txt_k, axis=1)
v = op.Concat(img_v, txt_v, axis=1)
attn_out = op.Attention(q, k, v, ...)
img_out, txt_out = op.Split(attn_out, axis=1, ...)
```

### AdaLN-Zero modulation

Most transformer denoisers use AdaLN-Zero: project timestep embedding to
6 parameters (shift, scale, gate for attention and FFN):

```python
# Modulation: timestep → SiLU → Linear → split into 6 chunks
mod = self.mod(op, temb)  # [B, 6 * dim]
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = op.Split(
    mod, num_outputs=6, axis=-1
)

# Apply: norm → scale/shift → attention → gate → residual
normed = self.norm(op, x)
modulated = op.Add(op.Mul(normed, op.Add(scale_msa, ONE)), shift_msa)
attn_out = self.attn(op, modulated)
x = op.Add(x, op.Mul(attn_out, gate_msa))
```

## Adding a new diffusion model — step by step

### 1. Create config class

In `_diffusers_configs.py`:

```python
@dataclasses.dataclass
class MyDenoiserConfig:
    # Parse from HF diffusers config.json
    in_channels: int = 4
    ...
    
    @classmethod
    def from_diffusers(cls, config: dict) -> "MyDenoiserConfig":
        return cls(...)
```

### 2. Create model class

In `models/my_model.py`:

```python
class MyDenoiser2DModel(nn.Module):
    default_task = "denoising"
    category = "Diffusion"
    config_class = MyDenoiserConfig
    
    def __init__(self, config: MyDenoiserConfig):
        super().__init__()
        # Build architecture matching HF naming
        ...
    
    def forward(self, op, sample, timestep, encoder_hidden_states):
        # Denoising forward pass
        ...
    
    def preprocess_weights(self, state_dict):
        # Ideally a no-op if naming matches HF
        return state_dict
```

### 3. Register in _DIFFUSERS_CLASS_MAP

In `_diffusers_builder.py`, add to `_init_diffusers_class_map()`:

```python
_DIFFUSERS_CLASS_MAP["MyDenoiser2DModel"] = (
    MyDenoiser2DModel, MyDenoiserConfig, "denoising"
)
```

### 4. Add task class (if needed)

If the existing `DenoisingTask` doesn't match the I/O signature, create a
new task in `tasks/`. Most denoisers use the standard `DenoisingTask`.

### 5. Add unit test

In `tests/build_graph_test.py`, add a tiny config:

```python
("my_denoiser", MyDenoiser2DModel, MyDenoiserConfig(
    in_channels=4, out_channels=4, ...  # Tiny values
), "denoising"),
```

### 6. Match HF weight names

Compare `named_parameters()` output with HF weight names. Use the
techniques from the **weight-name-alignment** skill to minimize
`preprocess_weights`.

## Naming conventions for diffusers models

Diffusers uses different conventions than transformers:

| Diffusers | Transformers |
|-----------|-------------|
| `GroupNorm` | `LayerNorm` / `RMSNorm` |
| `ResNet` blocks | Decoder layers |
| `down_blocks` / `up_blocks` | `layers` |
| `resnets` / `attentions` (within blocks) | `self_attn` / `mlp` |
| `to_q` / `to_k` / `to_v` / `to_out` | `q_proj` / `k_proj` / `v_proj` / `o_proj` |
| `conv_in` / `conv_out` | `embed_tokens` / `lm_head` |
| `time_embedding` | N/A |
| `encoder_hid_proj` | N/A |

## Reference files

| File | Contains |
|------|----------|
| `_diffusers_configs.py` | All diffusers config dataclasses |
| `_diffusers_builder.py` | Pipeline detection, `_DIFFUSERS_CLASS_MAP`, build functions |
| `models/unet.py` | UNet2DConditionModel + all UNet blocks |
| `models/vae.py` | AutoencoderKLModel + encoder/decoder blocks |
| `models/dit.py` | DiTTransformer2DModel + AdaLN blocks |
| `models/flux_sd3.py` | FluxTransformer2DModel + SD3Transformer2DModel |
| `models/controlnet.py` | ControlNetModel |
| `models/adapters.py` | T2IAdapterModel + IPAdapterModel |
| `models/qwen_image.py` | QwenImageTransformer2DModel |
| `models/qwen_image_vae.py` | AutoencoderKLQwenImageModel (3D causal VAE) |
| `models/video_vae.py` | VideoAutoencoderModel (3D video VAE) |
| `tasks/_denoising.py` | DenoisingTask |
| `tasks/_vae.py` | VAETask, QwenImageVAETask |
| `tasks/_controlnet.py` | ControlNetTask |
| `tasks/_adapter.py` | AdapterTask |
