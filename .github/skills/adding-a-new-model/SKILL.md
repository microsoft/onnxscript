---
name: adding-a-new-model
description: >
  Step-by-step guide for adding a new HuggingFace model architecture to the
  mobius package. Covers config extraction, model class creation,
  registry registration, weight preprocessing, and testing. Use this skill
  when the user wants to add support for a new model architecture (LLM,
  encoder-only, encoder-decoder, vision, audio, diffusion, or multimodal).
---

# Skill: Adding a New Model

## When to use

Use this skill when adding support for a new HuggingFace model architecture
(e.g. a new LLM family, vision model, encoder-decoder, audio model, or
diffusion component) to the `mobius` package.

## Prerequisites

- Identify the HuggingFace `model_type` string (from the model's `config.json`)
- Find a small checkpoint on HuggingFace Hub for testing
- Have the HuggingFace `transformers` source available to reference the
  PyTorch implementation

## Step-by-step

### 1. Check if the base `CausalLMModel` already works

Many models (LLaMA, Mistral, Qwen2, DeepSeek) use the standard decoder-only
architecture with no special components.  Before writing a custom class,
check whether `CausalLMModel` from `models/base.py` produces correct results:

```python
from mobius._registry import registry
from mobius.models.base import CausalLMModel

registry.register("my_model_type", CausalLMModel)
model = build("org/my-model-id", load_weights=True)
```

If the logits match HuggingFace, you only need the registry entry.

### 2. Create the model file

Create `src/mobius/models/<model_name>.py`.  The minimal template:

```python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Attention, DecoderLayer, Embedding, Linear, MLP, RMSNorm,
    create_attention_bias, initialize_rope,
)
from mobius.models.base import CausalLMModel


class MyTextModel(nn.Module):
    """Text model for MyArchitecture."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MyDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(self, op, input_ids, attention_mask, position_ids, past_key_values=None):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(op, input_ids=input_ids, attention_mask=attention_mask)

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op, hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class MyCausalLMModel(CausalLMModel):
    """Causal LM wrapper for MyArchitecture."""

    def __init__(self, config: ArchitectureConfig):
        # Skip CausalLMModel.__init__ to use custom TextModel
        nn.Module.__init__(self)
        self.config = config
        self.model = MyTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
```

#### Class metadata attributes

Every registered model class should set two class-level attributes used for
**auto-task detection** and **documentation generation**:

| Attribute | Type | Default (from `CausalLMModel`) | Description |
|-----------|------|------|-------------|
| `default_task` | `str` | `"text-generation"` | Task auto-selected by `build()` / CLI when no `--task` is specified. |
| `category` | `str` | `"Text Generation"` | Grouping label in the generated docs. |

The `_default_task_for_model()` function reads `default_task` from the
registered class, so there is **no hardcoded task map** to maintain.
The doc generator (`docs/_generate_models.py`) reads both attributes to
produce per-model pages and the categorised index automatically.

Override these when the model isn't a standard text-generation model:

```python
class MyMultiModalModel(nn.Module):
    """My vision-language model."""

    default_task: str = "vision-language"
    category: str = "Multimodal"
```

Standard categories: `"Text Generation"`, `"Mixture of Experts"`,
`"Multimodal"`, `"Speech-to-Text"`, `"Audio"`, `"Diffusion"`,
`"autoencoder"`, `"encoder-only"`, `"encoder"`, `"encoder-decoder"`,
`"vision"`, `"causal-lm"`.  New categories are picked up
automatically by the doc generator.

### 3. Identify what's different

Compare the HuggingFace PyTorch source against `CausalLMModel` / `DecoderLayer`.
Common variations to look for:

| Variation | Example | Solution |
|-----------|---------|----------|
| Custom norm (weight + 1) | Gemma | Subclass `RMSNorm` |
| Embedding scaling | Gemma (`* sqrt(d)`) | Subclass `Embedding` |
| Extra norms (pre/post feedforward) | Gemma2, Gemma3 | Custom `DecoderLayer` |
| QK normalization | Gemma3, Qwen3 | Set `attn_qk_norm=True` in config |
| Sliding window attention | Gemma2, Gemma3 | Alternating layer types + `sliding_window` config |
| Different activation | Various | Set `hidden_act` in config (handled by `MLP`) |
| Biased attention projections | Phi, PhiMoE | Set `attn_qkv_bias=True`, `attn_o_bias=True` |
| LayerNorm epsilon | Whisper (`1e-5`) | Pass eps from config to `LayerNorm(hidden_size, eps=...)` — default `1e-6` is wrong for many models |
| Q pre-scaling | Whisper | Multiply Q by `head_dim**-0.5` before Attention op, set `scale=1.0` in op |
| Causal self-attention (decoder) | Whisper | Set `is_causal=1` attribute on Attention op instead of explicit mask |
| Weight-free LayerNorm | OLMo-1B | Use `_WeightFreeLayerNorm` (constant scale=1, bias=0) — OLMo uses `F.layer_norm` with `weight=None, bias=None, eps=1e-5` |
| Custom attention scale | Granite | Pass `scale=config.attention_multiplier` to `Attention(config, scale=...)` instead of default `1/sqrt(head_dim)` |
| Embedding multiplier | Granite (`* 12.0`) | Multiply embeddings by `config.embedding_multiplier` in `TextModel.forward` after embed lookup |
| Logits scaling | Granite (`/ 8.0`) | Divide logits by `config.logits_scaling` in `CausalLMModel.forward` |
| Residual scaling | Granite (`* 0.22`) | Apply `residual + output * residual_multiplier` — **NOT** `residual * multiplier + output` |
| MoE layers | PhiMoE, GPTOSS | See the MoE skill |
| Vision encoder | Gemma3 | See the multimodal skill |
| Gated attention output | Qwen3.5 (`attn * sigmoid(gate)`) | Subclass `Attention` with doubled q_proj → Q+gate split |
| OffsetRMSNorm (1+weight) | Qwen3.5 | Use `OffsetRMSNorm` from `components/_rms_norm.py` |
| Hybrid layer types | Qwen3.5 (DeltaNet + full attention) | Use `config.layer_types` list to dispatch per-layer |
| Linear attention (DeltaNet) | Qwen3.5 | Use `GatedDeltaNet` component, stateless export |
| Shared expert with sigmoid gate | Qwen3.5-MoE | Custom `Qwen35MoEBlock` with `shared_expert_gate` Linear(hidden, 1) |
| Interleaved M-RoPE | Qwen3.5 | Use `InterleavedMRope` (not `ChunkedMRope`) |
| Fused QKV projection | ModernBERT | Split `Wqkv` [3H, H] into q/k/v in `preprocess_weights` |
| Fused gate+up (GeGLU/SwiGLU) | ModernBERT | Split `Wi` [2I, H] into gate/up in `preprocess_weights` |
| Pre-norm encoder with RoPE | ModernBERT | Custom encoder model — BERT is post-norm, CausalLM is causal |
| Hierarchical multi-scale | SAM2, Segformer | Per-stage embed dims/heads, dimension projection at stage boundaries |
| Efficient attention (seq reduction) | Segformer | Strided Conv2d on K/V to reduce sequence length before attention |
| DPT decoder (dense prediction) | Depth Anything | Multi-index backbone extraction + reassemble + fusion + prediction head |
| Detection tokens + heads | YOLOS | Learnable tokens appended to ViT sequence, MLP class/bbox heads |
| Subclass-only (weight rename) | BLIP, TrOCR, LayoutLMv3 | Override only `preprocess_weights` — lowest effort for compatible architectures |

### 4. Handle weight name mismatches (`preprocess_weights`)

If HuggingFace uses different weight names than your component tree, override
`preprocess_weights`:

```python
class MyCausalLMModel(CausalLMModel):
    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Example: rename keys
        renamed = {}
        for key, value in state_dict.items():
            new_key = key.replace("old_prefix.", "new_prefix.")
            renamed[new_key] = value
        # Call parent for weight tying
        return super().preprocess_weights(renamed)
```

**Common operations:**

- Strip prefixes: `language_model.model.X` → `X` (multimodal models)
- Rename expert weights: `w1` → `gate_proj` (MoE models)
- Weight tying: copy `embed_tokens.weight` → `lm_head.weight`
- Split fused QKV: `Wqkv.weight` [3H, H] → `q_proj`, `k_proj`, `v_proj` (ModernBERT)
- Split fused gate+up: `Wi.weight` [2I, H] → `gate_proj`, `up_proj` (ModernBERT)
- Split fused BLIP QKV: `in_proj_weight` [3H, H] → separate Q/K/V projections

**Tip:** Build the ONNX model, list its initializer names, and compare against
the HuggingFace state dict keys to find mismatches:

```python
model_names = set(onnx_model.graph.initializers.keys())
hf_names = set(state_dict.keys())
print("In HF but not model:", hf_names - model_names)
print("In model but not HF:", model_names - hf_names)
```

### 5. Register the model

Add to `_create_default_registry()` in `src/mobius/_registry.py`:

```python
from mobius.models import MyCausalLMModel

# In _create_default_registry():
reg.register("my_model_type", MyCausalLMModel)
```

Also export from `src/mobius/models/__init__.py`.

### 6. Update `ArchitectureConfig.from_transformers` if needed

If the model has unusual config fields, update `from_transformers()` in
`_configs.py`.  For example, nested RoPE configs or custom head-dim formulas.

### 7. Write tests

See the **writing-tests** skill for full details.  At minimum:

1. **Add a config entry to `tests/_test_configs.py`** in the appropriate
   group (`CAUSAL_LM_CONFIGS`, `ENCODER_CONFIGS`, `SEQ2SEQ_CONFIGS`,
   `VISION_CONFIGS`, or `DETECTION_CONFIGS`):
   ```python
   # In tests/_test_configs.py:
   CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
       # ...
       ("my_model_type", {"hidden_act": "gelu"}, True),
   ]
   ```
   - Set `is_representative=True` if the model has unique behaviour
     (custom model class, special config like softcapping, partial rotary,
     ALiBi, MoE, etc.)
   - Set `is_representative=False` if it's an alias of an existing base
     class with no special config overrides
   - Text-generation models with no special config are auto-generated from
     the registry, but explicit entries are preferred for documentation

   Then verify:
   ```bash
   pytest tests/build_graph_test.py -k "my_model_type"
   ```

2. **Add a small model to `tests/integration_test.py`** if a small HuggingFace
   checkpoint exists (< 1 B parameters preferred):
   ```python
   # In _TEXT_MODELS list:
   pytest.param("org/my-small-model", False, id="my-model"),
   ```

3. **Testing large models with random weights:**

   When the smallest available checkpoint is too large for CI (e.g. Qwen3.5
   at 27B), create a HF model with random weights and reduced layers:

   ```python
   from transformers import AutoConfig
   from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

   c = AutoConfig.from_pretrained("Qwen/Qwen3.5-27B")
   tc = c.text_config
   tc.num_hidden_layers = 4
   tc.layer_types = tc.layer_types[:4]  # Must match num_hidden_layers
   hf_model = Qwen3_5ForCausalLM._from_config(tc, dtype=torch.float32)
   ```

   Then use `build_from_module` with the HF state dict to compare logits.
   See `test_qwen35_prefill_logits_match` in `tests/integration_test.py`.

4. **Test with the CLI** to verify end-to-end build works:
   ```bash
   # Single model
   mobius build --model org/my-small-model mymodel/output

   # Multi-component (encoder-decoder) — produces separate encoder.onnx + decoder.onnx
   mobius build --model org/my-small-model mymodel/output
   ```
   The task is auto-detected from the model type. Use `--task` to override
   if needed (e.g. `--task text-generation`).

### 8. Documentation

Model documentation is **auto-generated** from class metadata by
`docs/_generate_models.py` at doc-build time.  No manual doc update is needed
if you set the `default_task`, `category`, and a good class docstring (the
first paragraph is used as the model description).

The README model list is a curated highlight table — update it only if the
new model is a significant addition.

## Checklist

- [ ] Model file in `src/mobius/models/` with Apache-2.0 copyright header
- [ ] Class has `default_task` and `category` attributes (if not standard text-generation)
- [ ] Class has a descriptive docstring (first paragraph used in generated docs)
- [ ] `preprocess_weights` handles any key mismatches
- [ ] Registered in `_create_default_registry()`
- [ ] Exported from `models/__init__.py`
- [ ] Config extraction works (`ArchitectureConfig.from_transformers`)
- [ ] Tiny config in `tests/_test_configs.py` (with `is_representative` flag)
- [ ] Integration test model in `tests/integration_test.py` (if small checkpoint available)
- [ ] CLI build works (`mobius build --model ...`)

**Note:** Default optimizer passes (CSE, deduplicate initializers, identity
elimination, remove unused nodes/opsets) are applied automatically by
`build_from_module`.

## Example: minimal diff for a LLaMA-compatible model

If the new model is fully LLaMA-compatible, the entire change is:

```python
# _registry.py
reg.register("my_llama_variant", CausalLMModel)
```

No new model file needed.

## Example: adding a non-LLM model

For models that aren't causal LMs, follow the same steps but use the
appropriate base class and task:

| Model type | Base class / pattern | Task | Config |
|------------|---------------------|------|--------|
| Encoder-only (BERT-like) | `BertModel` | `feature-extraction` | `ArchitectureConfig` |
| Encoder-only (ModernBERT) | `ModernBertModel` | `feature-extraction` | `ArchitectureConfig` |
| Encoder-decoder (BART/T5-like) | `BartForConditionalGeneration` or `T5ForConditionalGeneration` | `seq2seq` | `ArchitectureConfig` |
| Vision (ViT-like) | `ViTModel` or `CLIPVisionModel` | `image-classification` | `ArchitectureConfig` |
| Object detection | `YolosForObjectDetection` | `object-detection` | `ArchitectureConfig` |
| Depth estimation | `DepthAnythingForDepthEstimation` | `image-classification` | `ArchitectureConfig` |
| Segmentation | `SegformerForSemanticSegmentation` or `Sam2VisionModel` | `image-classification` | `ArchitectureConfig` |
| Audio encoder (Wav2Vec2-like) | `Wav2Vec2Model` | `audio-feature-extraction` | `ArchitectureConfig` |
| Multimodal (LLaVA-like) | `LLaVAModel` | `vision-language` | `ArchitectureConfig` |
| Document AI | `LayoutLMv3Model` | `feature-extraction` | `ArchitectureConfig` |
| OCR decoder | `TrOCRForConditionalGeneration` | `seq2seq` | `ArchitectureConfig` |
| Diffusion denoiser | Custom (`UNet2DConditionModel`, etc.) | `denoising` | Custom config (e.g. `UNet2DConfig`) |
| VAE | `AutoencoderKLModel` | `vae` | `VAEConfig` |
| Adapter | `T2IAdapterModel` / `IPAdapterModel` | `adapter` | Custom config |

Many new models can be registered as aliases of existing classes (e.g.
`reg.register("my_bert_variant", BertModel)`) if the architecture matches.

## False Compatibility Pitfalls

When registering models as aliases of existing base classes, **tests passing
does not mean the mapping is correct.** Graph-build tests only check that an
ONNX graph can be constructed — they do NOT verify that the graph matches
the model's actual computation.

### Safe approximate mappings

The project accepts "approximate" registry aliases when the model uses
similar-but-not-identical attention. These produce structurally correct ONNX
graphs; weight-loading may need minor adjustments:

| Model | Maps to | Why it works |
|-------|---------|-------------|
| DeBERTa | `BertModel` | Disentangled attention is a variant of standard attention |
| Swin | `ViTModel` | Shifted window attention is still self-attention over patches |
| SqueezeBERT | `BertModel` | Grouped convolution replaces dense attention, but same I/O shape |

### NEVER safe as registry aliases

These model families have fundamentally different computation that **cannot**
be represented by standard base classes, even though `build_graph_test` passes:

| Category | Models | Why it fails |
|----------|--------|-------------|
| Pure CNNs | ConvNeXt, ResNet, MobileNet, EfficientNet, RegNet | No attention at all — base ViT/BERT classes produce attention-based graphs |
| Spatial pooling | PoolFormer | Uses spatial average pooling instead of attention — structurally incompatible |
| SSM / state-space models | Mamba, Mamba2, FalconMamba, RWKV, RecurrentGemma | Sequential scan / linear recurrence, not attention |
| Fundamentally different attention | Longformer (sparse), BigBird (block sparse), Funnel (downsampling) | Attention pattern differs from dense self-attention at a structural level |
| Custom tokenization | CANINE (character-level) | Byte-level input, hash embeddings — not a standard vocab embedding |

**Rule of thumb:** If the HuggingFace model's `forward()` method doesn't call
`self_attn(query, key, value)` in a standard way, it is NOT a safe alias.

### Future work

CI currently only runs graph-build tests (shape inference, op validity). To
catch false compatibility in approximate mappings, we need **weight-loading
tests** that:
1. Load real HuggingFace weights into the ONNX graph
2. Run inference on a test input
3. Compare output against HuggingFace PyTorch output
4. Fail if max abs diff exceeds a threshold (e.g. 0.01)

This would catch shape mismatches, wrong norm types, and missing scaling
factors that graph-build tests cannot detect.

## Troubleshooting: common pitfalls

This section documents real bugs found during model integration and how to
diagnose them.

### 1. ORT rejects graph with `tensor(double)` / wrong dtype

**Symptom:** `onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph` error
mentioning `Type 'tensor(double)'` for an input to a custom op like
`RotaryEmbedding`.

**Root cause:** NumPy creates float64 arrays by default when given Python
floats/lists.  If any intermediate computation uses float64, the result
propagates through the graph.

**Common culprit:** `LongRope` in `_rotary_embedding.py`:
```python
# BAD — creates float64 array, poisons cos/sin caches
long_factor = np.array(config.rope_scaling["long_factor"])

# GOOD — explicit float32
long_factor = np.array(config.rope_scaling["long_factor"], dtype=np.float32)
```

**Fix:** Always pass `dtype=np.float32` when creating numpy arrays from config
values.  Search for `np.array(` without an explicit dtype to find other
instances.

### 2. Normalization type mismatch (RMSNorm vs LayerNorm)

**Symptom:** Logits have large max abs diff (> 0.5) from HuggingFace, but
weight loading succeeds and the greedy token may still match.

**Diagnosis:** Check what norm class the HuggingFace model actually uses.
Many models have custom norm classes that differ from the standard:

```python
import inspect
from transformers.models.olmo.modeling_olmo import OlmoLayerNorm
print(inspect.getsource(OlmoLayerNorm))
```

**Known cases:**
- **OLMo-1B**: Uses `OlmoLayerNorm` which is `F.layer_norm(x, ..., weight=None, bias=None, eps=1e-5)` — a weight-free **LayerNorm**, NOT RMSNorm.  Solution: use `_WeightFreeLayerNorm` (constant scale=1, bias=0) from `models/olmo.py`.
- **Whisper**: Uses `LayerNorm` with `eps=1e-5`, not the default `1e-6`.
- **Gemma**: Uses `RMSNorm` but adds 1 to the weight before applying.

**Key difference between LayerNorm and RMSNorm:**
- **LayerNorm** subtracts the mean, then divides by std: `(x - mean) / sqrt(var + eps) * gamma + beta`
- **RMSNorm** does NOT subtract the mean: `x / sqrt(mean(x²) + eps) * gamma`

Using the wrong type causes systematic error that grows through layers.

### 3. Missing scaling multipliers

**Symptom:** Token generation diverges after a few tokens, but the first few
tokens may match.

**Diagnosis:** Check the HuggingFace model for multiplier/scaling attributes
that aren't in the standard Llama config:

```python
config = AutoConfig.from_pretrained("model-id")
for k, v in config.to_dict().items():
    if "multiplier" in k or "scaling" in k:
        print(f"{k}: {v}")
```

**Known cases:**
- **Granite**: Has four multipliers — `embedding_multiplier`, `attention_multiplier`, `logits_scaling`, `residual_multiplier`.  All must be implemented for correct results.

**Critical: residual scaling direction matters.**  Granite uses:
```python
# CORRECT: multiplier on the output, not the residual
hidden_states = residual + attn_output * residual_multiplier

# WRONG: multiplier on the residual
hidden_states = residual * residual_multiplier + attn_output
```

Always verify the direction by reading the HuggingFace source:
```python
from transformers.models.granite.modeling_granite import GraniteDecoderLayer
import inspect
print(inspect.getsource(GraniteDecoderLayer.forward))
```

### 4. Config fields not extracted

**Symptom:** Model builds without error but multipliers/special features are
not applied (they default to 1.0/None/False).

**Fix:** Add extraction to `ArchitectureConfig.from_transformers()` in
`_configs.py`:

```python
# In _configs.py ArchitectureConfig dataclass:
embedding_multiplier: float = 1.0
attention_multiplier: float | None = None

# In from_transformers() options dict:
embedding_multiplier=getattr(config, "embedding_multiplier", 1.0),
attention_multiplier=getattr(config, "attention_multiplier", None),
```

**Tip:** Use safe defaults (1.0 for multipliers, None for optional features)
so existing models are unaffected.

### 5. Attention scale override

**Symptom:** Attention scores are wrong, causing gradual drift in generation.

**Diagnosis:** Some models override the default `1/sqrt(head_dim)` scale:

```python
# Check HuggingFace attention class
from transformers.models.granite.modeling_granite import GraniteAttention
print(GraniteAttention.__init__)  # Look for self.scaling = ...
```

**Fix:** Use the `scale` parameter on the `Attention` component:

```python
# In your custom DecoderLayer:
self.self_attn = Attention(config, scale=config.attention_multiplier)
```

The `Attention.__init__` accepts an optional `scale: float | None` parameter.
When `None`, it defaults to `head_dim**-0.5`.

### 6. Weight-free norms (no learnable parameters)

**Symptom:** Weight keys like `model.norm.weight` appear in the model but not
in the HuggingFace state dict, and `preprocess_weights` fills them with ones.
The norm still uses the wrong algorithm (e.g. RMSNorm instead of LayerNorm).

**Fix:** Use `_WeightFreeLayerNorm` from `models/olmo.py` which creates
`nn.Parameter` with constant data (ones for scale, zeros for bias) that
the ONNX `LayerNormalization` op requires, but the HuggingFace model has no
corresponding weights for:

```python
class _WeightFreeLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(
            [hidden_size], data=ir.tensor(np.ones(hidden_size, dtype=np.float32))
        )
        self.bias = nn.Parameter(
            [hidden_size], data=ir.tensor(np.zeros(hidden_size, dtype=np.float32))
        )
        self.eps = eps

    def forward(self, op, hidden_states):
        return op.LayerNormalization(
            hidden_states, self.scale, self.bias, epsilon=self.eps, axis=-1
        )
```

### 7. Debugging workflow for logit mismatches

When ONNX logits don't match HuggingFace, follow this sequence:

1. **Check max abs diff on prefill** — tells you if the issue is in the
   model computation (not just autoregressive error accumulation):
   ```python
   diff = np.abs(onnx_logits[0, -1] - hf_logits)
   print(f"Max abs diff: {diff.max()}")  # > 0.01 is suspicious, > 0.5 is a bug
   ```

2. **Check the HuggingFace norm class** — most mismatches come from using the
   wrong normalization type or epsilon.

3. **Check for model-specific config fields** — inspect the HuggingFace config
   dict for any field with "multiplier", "scaling", "factor", "epsilon", or
   "bias" that isn't being extracted.

4. **Check weight dtype** — ensure numpy arrays in rope/embeddings use float32.

5. **Compare layer by layer** — if the diff is moderate (0.01-1.0), the issue
   is likely an architectural difference in the norm or residual pattern.  If
   the diff is huge (>10), check if weights are loaded to the wrong parameters.

### 8. Gated attention Q/gate split ordering

**Symptom:** Large logit diff (~2.0) on Qwen3.5 or similar gated attention models.

**Root cause:** HF splits Q and gate *within each head*: reshapes to
`[B, S, num_heads, 2*head_dim]` then chunks on last dim.  A naive midpoint
split of the flat tensor gives wrong results.

**Fix:** Reshape to per-head layout before splitting:
```python
# WRONG: split flat tensor at midpoint
q, gate = op.Split(qg_proj, num_outputs=2, axis=-1)

# CORRECT: reshape to per-head, then split
qg = op.Reshape(qg_proj, [0, 0, num_heads, 2 * head_dim])
q, gate = op.Split(qg, [head_dim, head_dim], axis=-1)
```

### 9. DeltaNet missing query scaling

**Symptom:** Linear attention output is orders of magnitude too large.

**Root cause:** After L2-normalizing Q and K, you still need a
`1/sqrt(key_head_dim)` scaling factor on the query, similar to standard
attention.

### 10. Extracting `last_hidden_state` before vs after norm

**Symptom:** Downstream model (e.g. code predictor, projection layer)
receives wrong hidden states. Prefill logits match HF exactly, but
generation diverges immediately.

**Root cause:** HuggingFace's `outputs.last_hidden_state` is the
**post-norm** hidden state (after RMSNorm/LayerNorm). If you extract
the hidden state before the final norm, downstream consumers get
pre-norm values. In single-model LLMs this doesn't matter (the lm_head
is after the norm). In multi-model pipelines (TTS, VLM), the
hidden state is passed to another model, so norm ordering is critical.

**Fix:** Always extract hidden state *after* the model's final norm:
```python
# WRONG: hidden_states before norm
hidden_states = decoder_output  # pre-norm
logits = lm_head(norm(hidden_states))  # logits correct, but...
return logits, hidden_states  # hidden_states is WRONG for downstream

# CORRECT: apply norm first, then use for both logits and output
hidden_states = norm(decoder_output)  # post-norm
logits = lm_head(hidden_states)
return logits, hidden_states  # hidden_states matches HF
```

### 11. Identity node folding renames initializers

**Symptom:** Weight loading fails — `preprocess_weights` maps to the
original parameter name (e.g. `code_predictor.stacked_codec_embedding`)
but the initializer in the ONNX graph has been renamed to something
like `v_code_predictor.Identity_174`.

**Root cause:** The IR optimizer folds `Identity(initializer)` by
removing the Identity node and renaming the initializer to the
output name. If you then set a custom name on the output (e.g.
`codec_embeddings.name = "codec_embeddings"`), the initializer gets
renamed to `codec_embeddings` — breaking weight loading.

**Fix:** Use `op.Identity()` to create a *real* Identity node between
the initializer and the graph output. Ensure the elimination pass
retains Identity nodes that feed graph outputs. This creates a separate
output value, so renaming the output doesn't affect the initializer:
```python
# In forward():
codec_embeddings = op.Identity(self.stacked_codec_embedding)
return logits, present_key_values, codec_embeddings

# In task (safe to rename — Identity separates the names):
codec_embeddings.name = "codec_embeddings"
graph.outputs.append(codec_embeddings)
```

### 12. `np.ascontiguousarray` promotes 0-d arrays to 1-d

**Symptom:** ONNX `Gather` axis-reducing semantics break — the output
has an extra dimension (e.g. `(1, vocab)` instead of `(vocab,)`).

**Root cause:** `np.ascontiguousarray(scalar_array)` promotes shape
`()` to `(1,)`. This changes `Gather(axis=0)` from axis-reducing
(scalar index) to axis-preserving (1-d index).

**Fix:** Guard against 0-d arrays:
```python
if v.ndim > 0:
    v = np.ascontiguousarray(v)
```

### 13. Multi-token prefill in code predictors

**Symptom:** Code predictor generates garbage. Prefill logits are
slightly off compared to HF.

**Root cause:** Some architectures (e.g. Qwen3-TTS code predictor) use
a **2-token prefill**: `concat(projected_hidden, embed(code_0))` as two
separate tokens through the transformer. Summing them into 1 token
changes attention patterns and all subsequent hidden states.

**Diagnosis:** Compare the inputs_embeds shape at step 0. If HF passes
`(batch, 2, hidden)` but your model uses `(batch, 1, hidden)`, the
attention context window is wrong.

**Fix:** Construct inputs_embeds externally to match HF's exact flow:
```python
# Step 0 (prefill): 2 tokens
inputs = np.concatenate([talker_hidden, embed(code_0)], axis=1)  # (1, 2, H)
# Steps 1+: 1 token
inputs = cp_embed[step-1, code_i, :].reshape(1, 1, -1)  # (1, 1, H)
```

### 14. Embedding table index off-by-one in multi-step generation

**Symptom:** Codes are plausible but audio quality is wrong. Codec sum
doesn't match HF.

**Root cause:** In multi-step code prediction, HF uses
`embed[step-1](code)` at generation step `step`, not `embed[step]`.
The off-by-one means every embedding lookup uses the wrong table.

**Fix:** Carefully trace HF's generation loop to determine which
embedding table index corresponds to which generation step. Write a
comparison script that checks individual embedding lookups match.

### 15. Codec sum uses output codes, not input codes

**Symptom:** codec_sum diverges from HF even though individual
embeddings weights are identical.

**Root cause:** The codec sum `Σ embed[i](code_{i+1})` uses the
*generated* (output) code at each step, not the input code. If the
model returns embeddings of the input codes, the sum is wrong.

**Fix:** Compute codec_sum externally using the codes actually
generated at each step:
```python
codec_sum = talker_embed(code_0)
for i in range(num_groups - 1):
    # codes[i+1] is the OUTPUT of code predictor step i
    codec_sum += cp_embed[i, codes[i + 1], :]
```

### 16. Precision-sensitive ops need fp32 upcast

**Symptom:** Type mismatch errors (`tensor(float) vs tensor(bfloat16)`)
when loading a model built with `--dtype bf16`, or numerical drift compared
to HuggingFace when running in fp16/bf16.

**Root cause:** Operations like `exp`, `softplus`, `sigmoid` (in gated norms),
and RMSNorm variance are numerically sensitive and must run in float32 to
match HuggingFace, which explicitly upcasts with `.float()` /
`.to(torch.float32)`.

**Two distinct problems:**

1. **Naive `CastLike` everywhere** — keeps everything in the model dtype
   (e.g. bf16), but `exp` overflows and the SSM state diverges.
2. **Naive `Cast(to=ir.DataType.FLOAT)` everywhere** — computes in fp32 but forgets to cast
   back, producing type mismatches with downstream bf16 ops.

**Correct pattern — upcast → compute → cast back:**
```python
# 1. Upcast to fp32 for the sensitive region
dt_f32 = op.Cast(dt, to=ir.DataType.FLOAT)
dt_f32 = op.Softplus(dt_f32)
a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
da = op.Exp(op.Mul(dt_4d, a_4d))  # all fp32 here
...
# 2. Cast back to input dtype at the boundary
y = op.CastLike(y_f32, x)
new_state = op.CastLike(new_state_f32, ssm_state)
```

**How to identify which ops need fp32:** Check the HuggingFace source for
`.float()` or `.to(torch.float32)` calls.  Each one marks an fp32 region
that the ONNX graph must replicate.

**Known fp32-required regions:**

| Region | HF evidence | ONNX pattern |
|--------|-------------|-------------|
| SSM recurrence (A, dt, exp, state) | `self.A_log.float()`, `hidden_states.float()`, `B.float()`, `C.float()` | `Cast(to=ir.DataType.FLOAT)` all inputs, `CastLike` output |
| GatedRMSNorm (SiLU + variance) | `hidden_states.to(torch.float32)`, `gate.to(torch.float32)` | Explicit fp32 for both, `CastLike` output |
| RMSNorm variance | `hidden_states.to(torch.float32)` | ONNX `RMSNormalization` handles via `stash_type=1` (default) |

**When fp32 upcast is NOT needed:**
- Linear projections (`MatMul`) — runtime handles mixed precision
- SiLU on conv output — HF keeps in model dtype
- Standard attention — ONNX `Attention` op handles precision internally

**Use `CastLike` for** parameters/constants that should match the *current*
compute dtype (which is fp32 inside an upcast region, or the model dtype
outside).  Use `Cast(to=ir.DataType.FLOAT)` to explicitly enter an fp32 region.

## Reference implementations

| Model | File | Key differences from base |
|-------|------|--------------------------|
| Granite | `models/granite.py` | 4 scaling multipliers, custom attention scale |
| OLMo-1B | `models/olmo.py` | Weight-free LayerNorm (not RMSNorm), eps=1e-5 |
| OLMo-2 | `models/olmo.py` | Post-norm decoder layers, QK full norm |
| Gemma | `models/gemma.py` | RMSNorm weight+1, embedding scaling |
| Whisper | `components/_whisper.py` | Q pre-scaling, LayerNorm eps=1e-5, is_causal attr |
| Phi3.5 | `components/_rotary_embedding.py` | LongRope with float32 factors |
| Qwen3.5 | `models/qwen.py` | Hybrid DeltaNet + full attention, gated GQA, OffsetRMSNorm, interleaved MRoPE |
| Qwen3.5-MoE | `models/qwen.py` | Same hybrid attention + MoE FFN with shared expert (sigmoid gate) |
| Qwen3-TTS | `models/qwen3_tts.py` | 4-model TTS split, 2-token code predictor prefill, small_to_mtp projection, Identity-exposed weights |
| **BLIP** | `models/blip.py` | Subclass of ViTModel — only `preprocess_weights` (fused QKV split, renaming) |
| **YOLOS** | `models/yolos.py` | ViT + detection tokens + DETR-style MLP heads. New `object-detection` task |
| **Depth Anything** | `models/depth_anything.py` | ViT backbone + DPT decoder (reassemble + fusion + depth head). Uses `ConvTranspose2d` |
| **Segformer** | `models/segformer.py` | Hierarchical 4-stage encoder, efficient attention (strided Conv2d on K/V), Mix-FFN with depthwise conv |
| **SAM2** | `models/sam2.py` | Hiera backbone (per-stage dim transitions, fused QKV attention) + FPN neck with top-down fusion |
| **LayoutLMv3** | `models/layoutlmv3.py` | Subclass of BertModel — only `preprocess_weights` (spatial embedding filtering) |
| **TrOCR** | `models/trocr.py` | Subclass of BartForConditionalGeneration — only `preprocess_weights` (`output_projection` rename) |
| **ModernBERT** | `models/modernbert.py` | Pre-norm encoder with RoPE + GeGLU + bidirectional attention. Fused QKV/Wi splitting. Both encoder and decoder variants |
| **Gemma3n** | `models/gemma3n.py` | AltUp predict/correct, Laurel low-rank, per-layer input gating, hybrid local/global attention |
| **Mllama** | `models/mllama.py` | Interleaved cross-attention decoder, tanh-gated residual, manual QK-norm |

## Reference Examples

When adding a new model, use these files as canonical references:

| Complexity | File | Why |
|---|---|---|
| **Minimal** — base class works, only weight mapping needed | `models/phi3.py` (38 lines) | Extends `CausalLMModel`, only overrides `preprocess_weights()` to split fused QKV and gate-up projections. Shows the simplest possible model addition. |
| **Minimal** — encoder subclass | `models/layoutlmv3.py` | Extends `BertModel`, only overrides `preprocess_weights()`. Same pattern for encoder-only models. |
| **Moderate** — custom components | `models/gemma.py` | Adds custom attention (soft-capping), custom MLP (GeGLU), and custom normalization. Good example of component subclassing. |
| **Complex** — multi-model architecture | `models/qwen3_tts.py` | 4-model TTS split with talker, code predictor, embedding, and speaker encoder sub-modules. Shows how to structure multi-model architectures. |
