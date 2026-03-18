# Design Document — mobius

## Overview

`mobius` generates ONNX models for generative AI architectures
using the [onnxscript](https://github.com/microsoft/onnxscript) `nn` API.
Rather than tracing or exporting a PyTorch model, it **constructs** the ONNX
graph directly from a declarative description of the architecture, then
applies pretrained HuggingFace weights.

### Key design goals

| Goal | Approach |
|------|----------|
| Correctness | Numerical parity with HuggingFace PyTorch models (< 1 % logit mismatch) |
| Modularity | Small, reusable components that compose into full models |
| Extensibility | New architectures require only a model file + registry entry |
| Testability | Unit tests per component; integration tests per architecture |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Public API                             │
│  build()  build_from_module()  apply_weights()  registry   │
└────────────┬───────────────────────────────┬───────────────┘
             │                               │
     ┌───────▼────────┐            ┌─────────▼────────┐
     │ ArchitectureConfig │        │    ModelTask      │
     │ (from_transformers) │       │ CausalLMTask      │
     │ DiffusersConfigs    │       │ VisionLanguageTask│
     │                    │        │ VAETask, etc.     │
     └───────┬────────────┘        └─────────┬────────┘
             │                               │
     ┌───────▼───────────────────────────────▼──────┐
     │               Model Modules                   │
     │  CausalLMModel · LLaVAModel · WhisperModel    │
     │  UNet2DModel · DiTModel · AutoencoderKLModel  │
     │  BertModel · T5Model · Wav2Vec2Model · …      │
     └───────┬──────────────────────────────────────┘
             │
     ┌───────▼───────────────────────────────────────┐
     │             Component Library                   │
     │  Attention · MLP · DecoderLayer · RMSNorm · …  │
     │  MoELayer · VisionModel · MultiModalProjectors │
     │  ConvBlock · TimeEmbedding · ALiBiAttention     │
     └─────────────────────────────────────────────────┘
```

---

## Core concepts

### 1. Components (`src/mobius/components/`)

Components are `onnxscript.nn.Module` subclasses that build ONNX sub-graphs.
Each component's `forward(op, ...)` method receives an `OpBuilder` and
constructs ONNX nodes imperatively:

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        self.weight = nn.Parameter([hidden_size], name="weight")
        self.eps_param = nn.Parameter([], name="eps")

    def forward(self, op, hidden_states):
        return apply_rms_norm(op, hidden_states, self.weight, self.eps_param)
```

Key components:

| Component | Purpose |
|-----------|---------|
| `Embedding` | Token embedding lookup |
| `Linear` | `MatMul` (+ optional `Add` for bias) |
| `RMSNorm` / `LayerNorm` | Pre-norm / post-norm layers |
| `Attention` | Multi-head / grouped-query attention with KV cache |
| `MLP` | Gate + up + down projections with activation |
| `DecoderLayer` | Pre-norm residual: LayerNorm → Attention → LayerNorm → MLP |
| `MoELayer` | Mixture-of-Experts with pluggable gate |
| `VisionModel` | SigLIP-style patch embedding + transformer encoder |
| Projectors | `Gemma3MultiModalProjector`, `MLPMultiModalProjector`, `LinearMultiModalProjector` |
| `InputMixer` | Scatter vision embeddings into text positions |
| RoPE variants | `DefaultRope`, `LinearRope`, `DynamicNTKRope`, `Llama3Rope`, `InterleavedMRope` |
| `ALiBiAttention` | Attention with linear biases (Falcon, Bloom) |
| `ConformerEncoder` | Conformer-based audio encoder (NeMo) |
| Diffusion blocks | `TimeEmbedding`, `CrossAttnDownBlock2D`, `UNetMidBlock2D`, `UpBlock2D` |

**Design principle:** Components are reusable across model families.  When a
model family has a unique variant (e.g. Gemma's `weight + 1` norm), create a
subclass rather than adding flags.

### 2. Models (`src/mobius/models/`)

A model module composes components into a full architecture.  The simplest
text model follows this template:

```
CausalLMModel
 ├── TextModel
 │    ├── Embedding (embed_tokens)
 │    ├── DecoderLayer × num_hidden_layers
 │    │    ├── RMSNorm (input_layernorm)
 │    │    ├── Attention (self_attn)
 │    │    ├── RMSNorm (post_attention_layernorm)
 │    │    └── MLP (mlp)
 │    ├── RMSNorm (norm)
 │    └── RoPE (rotary_emb)
 └── Linear (lm_head)
```

Model classes implement:

- `__init__(config)` — construct the component tree
- `forward(op, input_ids, attention_mask, position_ids, past_key_values)` — wire
  the computation graph
- `preprocess_weights(state_dict)` — rename / transform HuggingFace weights to
  match the component tree's parameter names

### 3. ArchitectureConfig (`src/mobius/_configs.py`)

A flat dataclass that normalises HuggingFace model configs into a uniform
schema.  `ArchitectureConfig.from_transformers(hf_config)` handles:

- Standard field extraction (hidden_size, num_layers, etc.)
- RoPE config normalisation (including nested `rope_scaling` dicts like Gemma3)
- MoE fields (`num_local_experts`, `num_experts_per_tok`)
- Vision fields (for multimodal models)
- Head-dim computation (explicit or `hidden_size / num_heads`)

All downstream code consumes `ArchitectureConfig`, never raw HF configs.

### 4. Model registry (`src/mobius/_registry.py`)

Maps `model_type` strings (from HuggingFace `config.json`) to module classes:

```python
registry.register("llama", CausalLMModel)
registry.register("gemma3_multimodal", Gemma3MultiModalModel)
```

`build(model_id)` uses the registry to auto-detect the correct class.

### 5. Tasks (`src/mobius/tasks/`)

A `ModelTask` defines the ONNX graph's input/output contract:

| Task | Inputs | Outputs |
|------|--------|---------|
| `CausalLMTask` | input_ids, attention_mask, position_ids, past_key_values.*.{key,value} | logits, present.*.{key,value} |
| `VisionLanguageTask` | (same as above) + pixel_values | (same as above) |
| `FeatureExtractionTask` | input_ids, attention_mask | hidden_states |
| `EncoderDecoderTask` | Encoder + decoder graphs (split components) | encoder: hidden_states; decoder: logits, present.*.{key,value} |
| `SpeechToTextTask` | (encoder-decoder) audio_features → text tokens | logits |
| `AudioFeatureExtractionTask` | audio_input | hidden_states |
| `VAETask` | latent (decoder), pixel_values (encoder) | decoded image / latent |
| `DenoisingTask` | sample, timestep, encoder_hidden_states | denoised output |
| `ControlNetTask` | sample, timestep, encoder_hidden_states, controlnet_cond | multi-scale features |
| `AdapterTask` | condition (T2I) or image_embeds (IP) | adapter features |
| `ImageClassificationTask` | pixel_values | class logits |

Tasks call the module's `forward()` method and wrap the result in a valid
ONNX model with proper input/output naming and types.

### 6. Weight pipeline

```
HuggingFace Hub
      │
      ▼
_download_weights(model_id)       ← downloads safetensors files
      │                              returns dict[str, torch.Tensor]
      ▼
module.preprocess_weights(sd)     ← renames / transforms keys
      │                              e.g. w1→gate_proj, strip prefixes
      ▼
apply_weights(onnx_model, sd)     ← matches keys to ONNX initializers
                                     uses ir.LazyTensor for dtype casting
```

**`preprocess_weights`** is the model-specific hook for mapping HuggingFace
state-dict keys to the component tree's parameter names.  Common operations:

- Weight tying: copy `embed_tokens.weight` → `lm_head.weight`
- Prefix stripping: `language_model.model.X` → `X` (multimodal)
- Expert renaming: `w1` → `gate_proj`, `w2` → `down_proj`, `w3` → `up_proj` (MoE)

---

## Model family patterns

### Standard text model (LLaMA, Mistral, Qwen2)

Use the generic `CausalLMModel` with `TextModel` + `DecoderLayer`.  No
customisation needed — the base components handle GQA, RoPE variants, and
optional biases via config flags.

### Gemma family

Subclass `RMSNorm` to add the `weight + 1` offset.  Subclass `Embedding` to
scale by `sqrt(hidden_size)`.  Gemma2 adds alternating sliding / full
attention layers.  Gemma3 adds four-norm decoder layers with QK norm.

### ALiBi models (Falcon, Bloom)

Use `FalconCausalLMModel` with `ALiBiAttention`.  ALiBi provides positional
information through attention biases rather than RoPE: slopes are
geometrically spaced as `2^(-8/num_heads * (h+1))` per head.

### MoE models (PhiMoE, GPTOSS)

Replace `MLP` with `MoELayer` in a custom `MoEDecoderLayer`.  The
`MoELayer` accepts a pluggable **gate**:

- `TopKGate`: Standard softmax + top-k routing (default)
- `SparseMixerGate`: Sequential expert selection with threshold masking (PhiMoE)

The `MoETextModel` accepts an optional `gate_factory` callable to inject
the correct gate variant per layer.

### Multimodal models (Gemma3, LLaVA, Phi3V)

`Gemma3MultiModalModel` composes:

1. `VisionModel` — SigLIP patch embedding + transformer encoder
2. `Gemma3MultiModalProjector` — AvgPool2d → RMSNorm → MatMul
3. `InputMixer` — scatter vision embeddings at image-token positions
4. `Gemma3CausalLMModel` — standard text decoder

`LLaVAModel` follows a similar pattern with MLPMultiModalProjector.
Many VL models (InternVL2, Pixtral, Idefics, Molmo, etc.) reuse the LLaVA
pattern with CLIP/SigLIP vision encoder + projector + LLM.

Uses `VisionLanguageTask` which adds `pixel_values` to the ONNX graph inputs.

Three projector variants are available for different model families:

| Projector | Architecture | Used by |
|-----------|-------------|---------|
| `Gemma3MultiModalProjector` | AvgPool2d → RMSNorm → MatMul | Gemma3 |
| `MLPMultiModalProjector` | Linear → GELU → Linear | LLaVA, Phi4MM |
| `LinearMultiModalProjector` | Single Linear | PaliGemma |

### Encoder-only models (BERT, RoBERTa)

`BertModel` and `DistilBertModel` use `FeatureExtractionTask`. BERT-family
models use absolute positional embeddings (learned, not RoPE) with
`BertEmbeddings` and `EncoderLayer` components.

### Encoder-decoder models (BART, T5, Whisper)

`BartForConditionalGeneration` and `T5ForConditionalGeneration` use
`EncoderDecoderTask` with split encoder/decoder components. Whisper is a
specialised encoder-decoder for speech with `SpeechToTextTask`.

### Audio encoder models (Wav2Vec2, HuBERT, WavLM)

`Wav2Vec2Model` uses a CNN feature extractor followed by a transformer
encoder. Uses `AudioFeatureExtractionTask`.

### Diffusion models

Several model families for image/video generation:

| Model | Class | Task | Description |
|-------|-------|------|-------------|
| VAE | `AutoencoderKLModel` | `vae` | Image encoder/decoder for latent diffusion |
| UNet | `UNet2DConditionModel` | `denoising` | Stable Diffusion denoiser with cross-attention |
| DiT | `DiTTransformer2DModel` | `denoising` | Diffusion Transformer with AdaLN-Zero |
| SD3 | `SD3Transformer2DModel` | `denoising` | MMDiT joint text-image attention blocks |
| Flux | `FluxTransformer2DModel` | `denoising` | Double-stream + single-stream transformer |
| ControlNet | `ControlNetModel` | `controlnet` | Conditioning adapter with zero-conv outputs |
| Video VAE | `VideoAutoencoder3DModel` | `vae` | 3D convolution autoencoder for video |

### Adapter models

| Model | Class | Task | Description |
|-------|-------|------|-------------|
| T2I-Adapter | `T2IAdapterModel` | `adapter` | Pixel unshuffle + conv blocks → multi-scale features |
| IP-Adapter | `IPAdapterModel` | `adapter` | Image projection → cross-attention tokens |

---

## Testing strategy

For a comprehensive map of the test infrastructure — including all test files,
markers, CI workflows, fixtures, utilities, and quick-reference commands — see
[Test Infrastructure Map](design/test-infrastructure-map.md).

### Unit tests (in-source `*_test.py`)

Every component and model module has co-located tests that verify:

- Parameter shapes and counts
- Graph construction (nodes are emitted)
- Op-type assertions (e.g. `count_op_type(graph, "MatMul") == 3`)

Run with: `pytest src/ -m "not integration"`

### Integration tests (`tests/`)

Compare ONNX model outputs against HuggingFace PyTorch reference:

| Test suite | Model | Checks |
|-----------|-------|--------|
| `integration_test.py` | Qwen2.5-0.5B | Prefill + decode logits |
| `generation_test.py` | Qwen2.5-0.5B | Greedy generation token match |
| `moe_integration_test.py` | Phi-tiny-MoE-instruct | Prefill + decode + generation |
| `multimodal_integration_test.py` | Gemma-3-4b-pt | Prefill with image + decode |
| `build_graph_test.py` | All 128 model types | Graph construction (unit tests) |

Integration tests are marked `@pytest.mark.integration` and excluded from
the default test run.  They download real model weights and run inference.

### Testing utilities (`src/mobius/_testing/`)

| Utility | Purpose |
|---------|---------|
| `create_test_builder()` | Fresh `OpBuilder` + graph for unit tests |
| `create_test_input()` | Add typed graph inputs |
| `make_config()` | Minimal `ArchitectureConfig` with sensible defaults |
| `OnnxModelSession` | Save-to-disk + ORT inference session wrapper |
| `torch_forward()` | HuggingFace single-step forward pass |
| `assert_logits_close()` | Numerical comparison with diagnostics |
| `OnnxGenerator` | Multi-step greedy decoding over ONNX model |

---

## File layout

```
src/mobius/
├── __init__.py              # Public API
├── _configs.py              # ArchitectureConfig (128 supported architectures)
├── _diffusers_configs.py    # VAEConfig, UNet2DConfig, DiTConfig, etc.
├── _registry.py             # model type → module class mapping
├── components/
│   ├── __init__.py          # Component exports
│   ├── _activations.py      # get_activation() → SiLU, GELU, etc.
│   ├── _attention.py        # Multi-head / GQA attention with KV cache
│   ├── _audio.py            # ConformerEncoder (NeMo subsampling, T5 bias, Conformer layers)
│   ├── _common.py           # Embedding, Linear, LayerNorm, create_attention_bias
│   ├── _decoder.py          # DecoderLayer (pre-norm residual block)
│   ├── _lora.py             # LoRALinear (base + per-adapter A/B/scale)
│   ├── _mlp.py              # Gate-up-down MLP
│   ├── _moe.py              # MoELayer, TopKGate, SparseMixerGate
│   ├── _multimodal.py       # Projectors + InputMixer
│   ├── _rms_norm.py         # RMSNorm (uses ONNX RMSNormalization op)
│   ├── _rotary_embedding.py # RoPE variants + initialize_rope()
│   └── _vision.py           # PatchEmbedding, VisionEncoder, VisionModel
├── models/
│   ├── __init__.py          # Model exports (35+ model classes)
│   ├── base.py              # TextModel, CausalLMModel
│   ├── adapters.py          # T2IAdapterModel, IPAdapterModel
│   ├── bart.py              # BartForConditionalGeneration
│   ├── bert.py / distilbert.py  # Encoder-only models
│   ├── controlnet.py        # ControlNetModel
│   ├── dit.py               # DiTTransformer2DModel
│   ├── falcon.py            # FalconCausalLMModel (ALiBi)
│   ├── flux_sd3.py          # SD3Transformer2DModel, FluxTransformer2DModel
│   ├── gemma.py / gemma2.py / gemma3.py
│   ├── llava.py             # LLaVAModel (generic VL pattern)
│   ├── moe.py               # MoE models (PhiMoE, GPTOSS)
│   ├── phi.py / qwen.py / chatglm.py / …
│   ├── t5.py                # T5ForConditionalGeneration
│   ├── unet.py              # UNet2DConditionModel
│   ├── vae.py               # AutoencoderKLModel
│   ├── video_vae.py         # VideoAutoencoder3DModel
│   ├── vit.py               # ViTModel
│   ├── wav2vec2.py          # Wav2Vec2Model
│   └── whisper.py           # WhisperForConditionalGeneration
├── tasks/
│   ├── __init__.py          # CausalLMTask, VisionLanguageTask, VAETask, etc.
│   ├── _adapter.py          # AdapterTask
│   ├── _audio_feature_extraction.py  # AudioFeatureExtractionTask
│   ├── _controlnet.py       # ControlNetTask
│   ├── _denoising.py        # DenoisingTask
│   └── _vae.py              # VAETask
├── rewrite_rules/
│   ├── __init__.py          # Rule exports
│   └── _packed_attention.py # PackedAttention rewrite rule
└── _testing/
    ├── __init__.py          # Test utilities
    ├── comparison.py        # assert_logits_close, assert_generation_match
    ├── ort_inference.py     # OnnxModelSession, OnnxGenerator
    └── torch_reference.py   # HF model loading + forward helpers
```
