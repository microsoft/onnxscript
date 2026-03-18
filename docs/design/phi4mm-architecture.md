# Phi-4-Multimodal-Instruct Architecture Breakdown

> This document covers the HuggingFace architecture of microsoft/Phi-4-multimodal-instruct and the current implementation status in mobius.

**Model**: `microsoft/Phi-4-multimodal-instruct`
**HuggingFace class**: `Phi4MMForCausalLM`
**model_type**: `"phi4mm"`
**Architecture**: `Phi4MMForCausalLM`

---

## 1. High-Level Architecture

Phi-4-multimodal is a **trimodal model** (vision + speech/audio + text) built on a Phi-3-style text decoder with:
- A **SigLIP vision encoder** for image understanding
- A **Conformer speech encoder** for audio understanding  
- **LoRA adapters** on the text decoder that are switched per modality (vision LoRA vs speech LoRA)
- **MLP projectors** to map vision/audio features into the text embedding space

### Component Hierarchy (HuggingFace)
```
Phi4MMForCausalLM
├── model (Phi4MMModel)
│   ├── embed_tokens (Embedding)           # Text token embeddings
│   ├── embed_tokens_extend                # Phi4MMImageAudioEmbedding
│   │   ├── image_embed (Phi4MMImageEmbedding)
│   │   │   ├── img_processor (SigLIP ViT)  # Vision encoder
│   │   │   ├── img_projection (Sequential: Linear→GELU→Linear)  # 4608→3072→3072
│   │   │   ├── glb_GN (Parameter)         # Global newline separator
│   │   │   └── sub_GN (Parameter)         # Sub-image newline separator
│   │   └── audio_embed (Phi4MMAudioEmbedding)
│   │       ├── encoder (ConformerEncoder)  # 24-block Conformer
│   │       └── audio_projection (ModuleDict)
│   │           ├── speech (Sequential: Linear→GELU→Linear)  # 1024→3072→3072
│   │           └── vision (Sequential: Linear→GELU→Linear)  # 1024→3072→3072
│   ├── layers[0..31] (Phi4MMDecoderLayer)
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── self_attn (Phi4MMAttention)
│   │   │   ├── qkv_proj (Linear, with LoRA A+B per adapter)
│   │   │   └── o_proj (Linear, with LoRA A+B per adapter)
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   └── mlp (Phi4MMMLP)
│   │       ├── gate_up_proj (Linear, with LoRA A+B per adapter)
│   │       └── down_proj (Linear, with LoRA A+B per adapter)
│   └── norm (RMSNorm)
└── lm_head (Linear, tied to embed_tokens)
```

### Component Hierarchy (ONNX — existing in codebase)
```
Phi4MMMultiModalModel
├── model (_Phi4MMMultiModalTextModel)
│   ├── embed_tokens (Embedding)
│   ├── embed_tokens_extend (_Phi4MMImageAudioEmbedding)
│   │   ├── image_embed (_Phi4MMImageEmbedding)
│   │   │   ├── img_processor (_Phi4MMSigLIPEncoder → VisionEncoder)
│   │   │   ├── img_projection (_Phi4MMProjectionMLP)
│   │   │   ├── glb_GN (Parameter)
│   │   │   └── sub_GN (Parameter)
│   │   └── audio_embed (_Phi4MMAudioEmbedding)
│   │       ├── encoder (ConformerEncoder)
│   │       └── audio_projection.speech (_Phi4MMProjectionMLP)
│   │       └── audio_projection.vision (_Phi4MMProjectionMLP)
│   ├── layers[0..N] (DecoderLayer with LoRALinear)
│   └── norm (RMSNorm)
└── lm_head (Linear)
```

---

## 2. Text Decoder Parameters

| Parameter | Value |
|---|---|
| `model_type` | `"phi4mm"` |
| `hidden_size` | 3072 |
| `intermediate_size` | 8192 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 8 (GQA: 3 groups) |
| `head_dim` | 128 (3072/24) |
| `vocab_size` | 200064 |
| `hidden_act` | `"silu"` (SwiGLU-style) |
| `rms_norm_eps` | 1e-5 |
| `max_position_embeddings` | 131072 |
| `original_max_position_embeddings` | 4096 |
| `rope_theta` | 10000.0 |
| `rope_scaling.type` | `"longrope"` |
| `partial_rotary_factor` | 0.75 |
| `rotary_ndims` | 96 (128 * 0.75) |
| `sliding_window` | 262144 |
| `attention_bias` | false |
| `mlp_bias` | false |
| `lm_head_bias` | false |
| `tie_word_embeddings` | true |
| `bos_token_id` | 199999 |
| `eos_token_id` | 199999 (gen config: [200020, 199999]) |
| `pad_token_id` | 199999 |

### Attention Mechanism
- **Grouped Query Attention (GQA)**: 24 query heads, 8 KV heads (ratio 3:1)
- Fused QKV projection: `qkv_proj` outputs `[q_size + kv_size + kv_size]` = `[3072 + 1024 + 1024]` = 5120
- **No bias** on attention projections
- **Partial RoPE**: Only 75% of head_dim (96 of 128 dims) gets rotary embedding; remaining 25% is passed through
- **LongRoPE** scaling with separate short_factor and long_factor arrays (each length 48 = rotary_ndims/2)

### MLP
- SwiGLU-style: `gate_up_proj` (fused gate+up, dim: 3072→16384), split into gate and up, then `silu(gate) * up`, then `down_proj` (8192→3072)
- **No bias**

### Decoder Layer
- Pre-norm architecture: `input_layernorm → self_attn → residual → post_attention_layernorm → mlp → residual`

---

## 3. Vision Encoder (SigLIP)

**Type**: SigLIP-SO400M/14 (NaViT variant, hardcoded in HF code)

| Parameter | Value |
|---|---|
| `hidden_size` | 1152 |
| `intermediate_size` | 4304 |
| `num_hidden_layers` | 27 |
| `num_attention_heads` | 16 |
| `head_dim` | 72 (1152/16) |
| `image_size` | 384 |
| `patch_size` | 14 |
| `hidden_act` | `"gelu_pytorch_tanh"` |
| `layer_norm_eps` | 1e-6 |
| Num patches per image | (384/14)² ≈ 729 (27×27), padded to 28×28=784 |

**Key details**:
- The SigLIP config is **NOT in config.json** — it's hardcoded in `get_siglip_vision_model()` (see `vision_siglip_navit.py`)
- Already captured in the ONNX codebase at `_configs.py` lines 393-398 with the hardcoded SigLIP values
- Uses `LayerNorm` (not RMSNorm)
- **No post_layernorm** — the ONNX implementation `_Phi4MMSigLIPEncoder` sets `post_layernorm=False`
- Has a learned position embedding (not sinusoidal): `position_embedding.weight` of shape `[num_patches, 1152]`
- The position embedding has odd sqrt (27×27=729), so there's a `ReflectionPad2d` to go from 27→28, giving 784 patches
- After AvgPool2d(2,2) compression: 28/2=14, so each crop produces 14×14=196 patch tokens
- **Layer index for features**: -2 (second-to-last hidden state)

### HD Transform (Dynamic High Resolution)
- `crop_size` = 448
- Images are split into multiple 448×448 crops (max 12 by default)
- Each crop → SigLIP → 196 tokens of dim 1152
- Global crop is always included (resized full image to 448×448)
- Sub-crops are arranged in a grid
- **Spatial merge**: 2×2 patches are merged → 4×1152=4608 dim features
- `glb_GN` and `sub_GN` are learnable separators inserted between global/sub image features and between rows
- Order: `sub_glb` (sub-images first, then global) — configured via `hd_transform_order`
- Total tokens per image: varies with resolution. Formula: `(h*w+1)*196 + 1 + (h+1)*14` where h,w are number of crop rows/cols

### Image Projection
- MLP: `Linear(4608, 3072) → GELU → Linear(3072, 3072)`
- Input dim = 4608 because of the 2×2 spatial merge (4 × 1152)
- Output dim = 3072 (text hidden_size)

---

## 4. Speech/Audio Encoder (Conformer)

**Type**: Cascades Conformer Encoder

| Parameter | Value |
|---|---|
| `attention_dim` | 1024 |
| `attention_heads` | 16 |
| `num_blocks` | 24 |
| `linear_units` | 1536 |
| `input_size` | 80 (mel filterbank features) |
| `kernel_size` | 3 |
| `activation` | `"swish"` |
| `time_reduction` | 8 |
| `causal` | true |
| `relative_attention_bias` | T5-style, max_distance=500 |
| `input_layer` | `"nemo_conv"` (conv_channels=1024) |
| `compression_rate` | 8 (from embd_layer config) |
| `downsample_rate` | 1 |

**Input**: 80-dim log mel filterbank features at 16kHz (10ms frame shift)
**Output**: 1024-dim per-frame features

### Audio Feature Extraction (Preprocessing)
- Input: Raw waveform (any format loadable by soundfile)
- Resample to 16kHz if needed
- Extract 80-dim log mel filterbank features using SpeechLib-compatible extraction
- Frame: 25ms window, 10ms shift (400 samples window, 160 hop at 16kHz)
- FFT size: 512
- Preemphasis: 0.97

### Audio Projection
- **Two separate projections** (ModuleDict with keys "speech" and "vision"):
  - `speech`: `Linear(1024, 3072) → GELU → Linear(3072, 3072)` — for speech-to-text tasks
  - `vision`: `Linear(1024, 3072) → GELU → Linear(3072, 3072)` — for vision-speech combined tasks
- Output dim = 3072 (text hidden_size)

### Audio Embed Size Calculation
- `compression_rate = 8` → after Conformer's time reduction (8×), the number of audio tokens is approximately `audio_frames / (compression_rate * feat_stride)`
- `feat_stride = 8` (from time_reduction)

---

## 5. LoRA Adapters

The text decoder uses **PEFT LoRA** with two named adapters that are activated depending on the input modality:

### Vision LoRA
| Parameter | Value |
|---|---|
| `r` (rank) | 256 |
| `lora_alpha` | 512 |
| Scale | 2.0 (alpha/r) |
| `lora_dropout` | 0.0 |
| Target modules | `layers.*((self_attn\.(qkv_proj\|o_proj))\|(mlp\.(gate_up\|down)_proj))` |

### Speech LoRA
| Parameter | Value |
|---|---|
| `r` (rank) | 320 |
| `lora_alpha` | 640 |
| Scale | 2.0 (alpha/r) |
| `lora_dropout` | 0.01 |
| Target modules | `((layers.*self_attn\.(qkv\|o)_proj)\|(layers.*mlp\.(gate_up\|down)_proj))` |

**Both adapters target the same layers** (all attention and MLP projections in all 32 decoder layers).

### LoRA Weight Names (HuggingFace)
For each targeted layer, HF stores:
- `model.layers.{i}.self_attn.qkv_proj.base_layer.weight` (base weight, wrapped by LoRA)
- `model.layers.{i}.self_attn.qkv_proj.lora_A.vision.weight` (rank 256)
- `model.layers.{i}.self_attn.qkv_proj.lora_B.vision.weight` (rank 256)
- `model.layers.{i}.self_attn.qkv_proj.lora_A.speech.weight` (rank 320)
- `model.layers.{i}.self_attn.qkv_proj.lora_B.speech.weight` (rank 320)
- (same pattern for o_proj, gate_up_proj, down_proj)

### Modality Switching
In HuggingFace, the model calls `set_lora_adapter("vision")` or `set_lora_adapter("speech")` to select which LoRA adapter is active. The processor sets `audio_projection_mode` to `"speech"` or `"vision"` depending on the input mode.

---

## 6. Special Tokens and Input Format

### Special Token IDs
| Token | ID | Name |
|---|---|---|
| Image placeholder | 200010 | `<\|endoftext10\|>` |
| Audio placeholder | 200011 | `<\|endoftext11\|>` |
| BOS/EOS/PAD | 199999 | |
| EOS (generation) | 200020 | `<\|end\|>` |

### Chat Template
```
<|system|>You are a helpful assistant.<|end|>
<|user|><|image_1|>Describe the image.<|end|>
<|assistant|>
```

### Placeholder Token Replacement Flow
1. Processor replaces `<|image_N|>` → repeated `<|endoftext10|>` (token ID 200010), count = num_img_tokens for that image
2. Processor replaces `<|audio_N|>` → repeated `<|endoftext11|>` (token ID 200011), count = audio_embed_size
3. In `embed_tokens_extend.forward()`:
   - Text tokens → `embed_tokens(input_ids)` → text embeddings
   - Positions where `input_ids == 200010` are replaced with image embedding features
   - Positions where `input_ids == 200011` are replaced with audio embedding features
   - The replacement uses `index_put` (non in-place)

### InputMode Enum
```python
class InputMode(Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3
```

---

## 7. Data Flow for Each Modality

### Text-Only
```
input_ids → embed_tokens → decoder_layers (no LoRA active) → norm → lm_head → logits
```

### Vision
```
image → Phi4MMImageProcessor (crops + resize + normalize)
      → input_image_embeds: [num_images, max_crops, 3, 448, 448]
      → SigLIP encoder → hidden_states[-2] → patch features: [N_crops, 784, 1152]
      → AvgPool2d(2,2) → [N_crops, 196, 1152]
      → 2×2 spatial merge → [N_crops, 196/4, 4608] (merge is complex: see HD transform)
      → HD transform with glb_GN/sub_GN separators
      → img_projection (Linear(4608,3072) → GELU → Linear(3072,3072))
      → Replace image token positions in text embeddings
      → decoder_layers (vision LoRA active) → norm → lm_head → logits
```

### Speech
```
audio → Phi4MMAudioFeatureExtractor (mel spectrogram, 80-dim)
      → input_audio_embeds: [num_audios, T_max, 80]
      → ConformerEncoder → [num_audios, T_reduced, 1024]
      → audio_projection["speech"] (Linear(1024,3072) → GELU → Linear(3072,3072))
      → Replace audio token positions in text embeddings
      → decoder_layers (speech LoRA active) → norm → lm_head → logits
```

### Vision + Speech
```
Both image and audio processed in parallel:
  - Image → SigLIP → img_projection → image_hidden_states
  - Audio → Conformer → audio_projection["vision"] (!) → audio_hidden_states
  NOTE: When vision+speech combined, audio uses the "vision" projection branch

Then merged:
  hidden_states = image_hidden_states * image_position_mask 
                + audio_hidden_states * non_image_position_mask
  → decoder_layers → norm → lm_head → logits
```

---

## 8. ONNX Implementation Status (Local Codebase)

### What EXISTS:
| Component | File | Status |
|---|---|---|
| `Phi4MMMultiModalModel` | `models/phi.py:439` | ✅ Defined |
| `Phi4MMCausalLMModel` | `models/phi.py:192` | ✅ Defined (text-only with LoRA) |
| `_Phi4MMSigLIPEncoder` | `models/phi.py:220` | ✅ Defined |
| `_Phi4MMImageEmbedding` | `models/phi.py:280` | ✅ Defined |
| `_Phi4MMAudioEmbedding` | `models/phi.py:299` | ✅ Defined |
| `_Phi4MMImageAudioEmbedding` | `models/phi.py:336` | ✅ Defined |
| `_Phi4MMMultiModalTextModel` | `models/phi.py:359` | ✅ Defined |
| `_Phi4MMProjectionMLP` | `models/phi.py:258` | ✅ Defined |
| `_LoRATextModel` | `models/phi.py:80` | ✅ Defined |
| LoRALinear component | `components/_lora.py` | ✅ Exists |
| ConformerEncoder component | `components/` | ✅ Exists |
| VisionEncoder component | `components/` | ✅ Exists |
| MultiModalTask | `tasks/_multimodal.py` | ✅ Defined |
| Registry entry | `_registry.py:487` | ✅ Registered ("phi4mm" + "phi4_multimodal") |
| Config extraction | `_configs.py:393` | ✅ SigLIP params hardcoded |
| Weight preprocessing | `models/phi.py:101` | ✅ `_preprocess_phi4mm_weights` |

### Key Architecture Decisions in ONNX Builder:
1. **Single-model output**: The `MultiModalTask` produces a single ONNX model with all components (vision encoder, audio encoder, text decoder) — NOT split into separate models like VisionLanguageTask
2. **LoRA as explicit weights**: LoRA A/B matrices are kept as separate ONNX parameters; the `LoRALinear` component computes `base(x) + scale * B(A(x))` during forward
3. **Fused QKV splitting**: Done in `preprocess_weights` — splits `qkv_proj` into separate `q_proj`, `k_proj`, `v_proj`
4. **Gate+Up splitting**: Done in `preprocess_weights` — splits `gate_up_proj` into separate `gate_proj` and `up_proj`
5. **LoRA weight rename**: Strips `base_layer.` prefix from LoRA-wrapped weights in `preprocess_weights`

---

## 9. Critical Details for Implementation

### Partial RoPE
- `partial_rotary_factor = 0.75`
- `rotary_ndims = int(128 * 0.75) = 96`
- Q/K are split: first 96 dims get RoPE, last 32 dims pass through unchanged
- This is the same as Phi-3

### LongRoPE Scaling
- `rope_scaling.type = "longrope"`
- Two factor arrays: `short_factor` (all 1.0s) and `long_factor` (varying, up to ~47.77)
- Each array has 48 elements (= rotary_ndims / 2 = 96 / 2)
- Scaling factor: `sqrt(1 + log(scale) / log(original_max_position_embeddings))` where `scale = max_pos / original_max_pos`
- Applied to cos/sin after frequency computation

### Tied Weights
- `tie_word_embeddings = true` → `lm_head.weight = embed_tokens.weight`
- The existing `_preprocess_phi4mm_weights` handles this by copying embed weight to lm_head

### Vision Position Embedding Shape
- HF stores as `[num_patches, hidden]` (2D)
- ONNX builder needs `[1, num_patches, hidden]` (3D) — handled in `preprocess_weights`

### Audio Projection Selection
- For speech-only: uses `audio_projection["speech"]`
- For vision+speech: uses `audio_projection["vision"]`
- This is a **runtime decision** in HF, but the ONNX builder currently only wires the speech projection

### Image Token Count
The number of image tokens per image is computed by the processor and is variable:
```python
# For each image:
num_tokens = 256 + 1 + int(mask.sum()) + int(mask[:,0].sum()) + 16
# 256 = 16×16 global tokens after spatial merge
# 1 = glb_GN separator
# mask.sum() = valid sub-image tokens after spatial merge
# mask[:,0].sum() = number of sub_GN row separators
# 16 = extra tokens from global image row separators
```

---

## 10. Weight Name Mapping Summary

### LoRA Weight Pattern
```
HF: model.layers.{i}.self_attn.qkv_proj.base_layer.weight
    model.layers.{i}.self_attn.qkv_proj.lora_A.{adapter}.weight
    model.layers.{i}.self_attn.qkv_proj.lora_B.{adapter}.weight

ONNX: model.layers.{i}.self_attn.qkv_proj.weight  (stripped "base_layer.")
      model.layers.{i}.self_attn.qkv_proj.lora_A.{adapter}.weight
      model.layers.{i}.self_attn.qkv_proj.lora_B.{adapter}.weight
```

### Vision Encoder Weight Pattern
```
HF: model.embed_tokens_extend.image_embed.img_processor.encoder.layers.{i}...
ONNX: model.embed_tokens_extend.image_embed.img_processor.encoder.layers.{i}...
(Direct alignment — matches well)
```

### Audio Encoder Weight Pattern
```
HF: model.embed_tokens_extend.audio_embed.encoder.blocks.{i}...
ONNX: model.embed_tokens_extend.audio_embed.encoder.blocks.{i}...
(Direct alignment)
```

### Projection Weight Pattern
```
HF: model.embed_tokens_extend.image_embed.img_projection.0.weight  (Linear)
    model.embed_tokens_extend.image_embed.img_projection.0.bias
    model.embed_tokens_extend.image_embed.img_projection.2.weight  (Linear after GELU)
    model.embed_tokens_extend.image_embed.img_projection.2.bias

HF: model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight
    model.embed_tokens_extend.audio_embed.audio_projection.speech.2.weight
    model.embed_tokens_extend.audio_embed.audio_projection.vision.0.weight
    model.embed_tokens_extend.audio_embed.audio_projection.vision.2.weight
```

---

## Implementation Status

*Summarized from codebase analysis (2026-03-09).*

### What's Implemented (✅)

**Model Architecture** — `src/mobius/models/phi.py`:

| Class | Purpose |
|-------|---------|
| `_LoRATextModel` | LoRA-aware text model (wraps decoder layers) |
| `Phi4MMCausalLMModel` | Text-only Phi4MM with LoRA (no vision/audio) |
| `_Phi4MMSigLIPEncoder` | SigLIP vision encoder (no post_layernorm) |
| `_Phi4MMProjectionMLP` | Linear→GELU→Linear projection |
| `_Phi4MMImageEmbedding` | Image embedding: SigLIP + projection + HD params |
| `_Phi4MMAudioEmbedding` | Audio embedding: ConformerEncoder + projection |
| `_Phi4MMImageAudioEmbedding` | Combined image+audio embedding container |
| `_Phi4MMMultiModalTextModel` | Text embedding + InputMixer for image/audio fusion |
| `Phi4MMMultiModalModel` | Main multimodal class: forward() + preprocess_weights() |

**Task**: `MultiModalTask` in `tasks/_multimodal.py` — single unified ONNX model with inputs for text, vision, and audio. Returns `ModelPackage({"model": ir.Model})`.

**Registry**: Registered as both `"phi4mm"` and `"phi4_multimodal"` pointing to `microsoft/Phi-4-multimodal-instruct`.

**Config Extraction**: Hardcoded SigLIP vision params in `_configs.py` (hidden_size=1152, 27 layers, patch_size=14). Audio config extracted from HF config.

**Weight Preprocessing**: Strips `.base_layer.` from LoRA keys, splits fused `qkv_proj`→q/k/v, splits fused `gate_up_proj`→gate/up, duplicates LoRA A weights, squeezes vision position embedding.

**Tests**: Unit tests in `build_graph_test.py` (graph construction, LoRA initializers, alias resolution). Integration tests in `phi4mm_integration_test.py` for all 4 modality combinations (text-only, vision, audio, vision+audio) — marked `@pytest.mark.integration_slow`, numerical parity unverified.

**Audio Infrastructure**: ConformerEncoder and related components in `components/_audio.py`. SpeechLanguageTask available for 3-model ASR split.

### What's Missing (❌)

1. **No end-to-end example** — `examples/` has nothing for phi4mm. Closest is `multimodal_generation.py` (Gemma3, no audio).
2. **No ORT-GenAI genai_config support** — `auto_export.py` and `GenaiConfigGenerator` don't handle `MultiModalTask` or audio inputs. `with_vision()` exists but no `with_audio()`.
3. **Hardcoded vision config** — SigLIP params baked into `_configs.py` instead of a proper config subclass.

### Architecture Comparison

| Model | Task Type | Models Produced | Vision | Audio |
|-------|-----------|-----------------|--------|-------|
| Gemma3 | `vision-language` | 3 (decoder, vision, embedding) | SigLIP | ❌ |
| Qwen2.5-VL | `qwen-vl` | 3 (decoder, vision, embedding) | ViT | ❌ |
| Qwen3 ASR | `speech-language` | 3 (audio_encoder, embedding, decoder) | ❌ | Conformer |
| **Phi-4 MM** | **`multimodal`** | **1 (unified model)** | **SigLIP** | **Conformer** |

Phi4MM uses a single unified model because LoRA adapters are shared across the full model, both encoders feed into the same InputMixer, and the architecture doesn't cleanly decompose into separate encoder/embedding/decoder.
