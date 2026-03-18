---
name: reusable-components
description: >
  Guide to the mobius component library and how to create or extend
  reusable building blocks (Attention, MLP, RMSNorm, RoPE, etc.). Covers
  parameter naming, ONNX op patterns, design principles (subclass over flags,
  model-agnostic components). Use this skill when creating new components or
  understanding the component architecture.
---

# Skill: Reusable Components

## When to use

Use this skill when creating or extending the building blocks that models are
composed from — attention layers, MLPs, normalisations, embeddings, RoPE
variants, and activations.

## Component library overview

All components live in `src/mobius/components/` and inherit from
`onnxscript.nn.Module`.  Each component's `forward(op, ...)` method builds
ONNX nodes via the `OpBuilder`.

```
components/
├── _activations.py       # get_activation(), SiLU module
├── _attention.py          # Multi-head / GQA attention with KV cache; Qwen35Attention (gated GQA)
├── _audio.py              # ConformerEncoder (NeMo subsampling, T5 bias, Conformer layers)
├── _common.py             # Embedding, Linear, LayerNorm, LayerNormNoAffine, GroupNorm, create_attention_bias
├── _conv.py               # Conv2d (2D convolution with bias and groups)
├── _decoder.py            # DecoderLayer (pre-norm residual block)
├── _encoder.py            # BertEmbeddings, EncoderAttention, EncoderLayer
├── _lora.py               # LoRALinear (base + per-adapter A/B/scale)
├── _mlp.py                # Gate-up-down MLP
├── _moe.py                # MoELayer, TopKGate, SparseMixerGate
├── _multimodal.py         # Projectors + InputMixer
├── _qwen3_vl_vision.py    # Qwen3-VL block-diagonal vision encoder
├── _gated_deltanet.py     # GatedDeltaNet (recurrent linear attention for Qwen3.5 hybrid)
├── _rms_norm.py           # RMSNorm, OffsetRMSNorm (1+weight), GatedRMSNorm (norm * SiLU gate)
├── _rotary_embedding.py   # RoPE variants (Default, Linear, Dynamic, Llama3, InterleavedMRope, ChunkedMRope)
├── _vision.py             # PatchEmbedding, VisionEncoder, VisionModel
└── _whisper.py            # Conv1d, WhisperAttention, WhisperDecoderLayer, WhisperEncoderLayer
```

Model files import shared primitives from `components/` and alias them with
an underscore prefix for local use:

```python
from mobius.components import Conv2d as _Conv2d, SiLU as _SiLU
```

Model-specific compound blocks (e.g. `_TimestepEmbedding`, `_DiTBlock`,
`_ResNetBlock2D`) remain in the model files they belong to.

## How to create a new component

### 1. Define the class

```python
from onnxscript import nn
from onnxscript._internal import builder


class MyComponent(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Name is automatically "weight" from the attribute name
        self.weight = nn.Parameter([hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states):
        # Build ONNX ops
        return op.Mul(hidden_states, self.weight)
```

### 2. Parameter naming

Parameter names are **automatically set** from the attribute name by
`nn.Module.__setattr__`.  You do **not** need to pass `name=` when the
attribute name matches the desired ONNX name:

```python
# GOOD — name is automatically "weight"
self.weight = nn.Parameter([hidden_size])

# Only use name= when the attribute name differs from the desired ONNX name
self.patch_embedding = nn.Parameter(
    [out_ch, in_ch, kH, kW], name="patch_embedding.weight"
)
```

When the component is nested in a module tree, names are automatically
prefixed by parent attribute names:

```python
# In model: self.layer = MyComponent(...)
# Resulting ONNX name: "layer.weight"
```

**Critical:** Parameter names must be unique within a component.  If two
parameters share the same attribute name at different levels, one will
silently overwrite the other.

To create a parameter with precomputed data (e.g. frozen positional embeddings),
use the `data=` argument:

```python
import onnx_ir as ir
self.embed_positions = nn.Parameter(
    [max_positions, d_model],
    name="embed_positions.weight",
    data=ir.tensor(numpy_array),
)
```

Do **not** assign `_const_value` directly.

### 3. Export from `__init__.py`

Add to `src/mobius/components/__init__.py`:

```python
__all__ = [..., "MyComponent"]
from mobius.components._my_component import MyComponent
```

### 4. Write unit tests

Create `_my_component_test.py` alongside the source:

```python
from mobius._testing import create_test_builder, create_test_input

class TestMyComponent:
    def test_forward(self):
        comp = MyComponent(hidden_size=64)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, 64])
        result = comp(op, x)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = MyComponent(hidden_size=64)
        names = [n for n, _ in comp.named_parameters()]
        assert "weight" in names
```

## Component reference

### Attention

```python
Attention(config)
Attention(config, scale=0.015625)  # Override default 1/sqrt(head_dim) scale
# Inputs:  hidden_states, attention_bias, position_embeddings, past_key_value
# Outputs: attn_output, (key_cache, value_cache)
```

Handles MHA, GQA, and MQA via `num_key_value_heads`.  Supports optional QK
norm (`attn_qk_norm=True`) and bias on Q/K/V/O projections.

The optional `scale` parameter overrides the default `head_dim**-0.5` attention
scale.  Use this when a model specifies a custom attention multiplier (e.g.
Granite's `attention_multiplier`).  When `None` (default), uses `1/sqrt(head_dim)`.

The ONNX `Attention` op (opset 23) has an `is_causal` attribute.  For
decoder self-attention in encoder-decoder models (e.g., Whisper), set
`is_causal=1` instead of building an explicit causal mask with
`create_attention_bias`.

Some models (Whisper) require **Q pre-scaling** for numerical parity with
HuggingFace: multiply Q by `head_dim**-0.5` before passing to `op.Attention`
and set `scale=1.0`.  This matches HF's order of operations and avoids
floating-point divergence in softmax.

**Qwen35Attention** (`_attention.py`): Gated GQA variant for Qwen3.5. Doubles
the Q projection to produce both Q and a gate signal, applies per-head
`OffsetRMSNorm` to Q and K, supports partial RoPE, and gates the output with
`attn_output * sigmoid(gate)`.

### MLP

```python
MLP(config)
# Uses gate_proj + up_proj + down_proj with configurable activation
```

The activation function comes from `config.hidden_act` and is resolved by
`get_activation()`.

### DecoderLayer

```python
DecoderLayer(config)
# Pre-norm residual: LayerNorm → Attention → Add → LayerNorm → MLP → Add
```

To customise, subclass and override the components:

```python
class MyDecoderLayer(DecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        # Replace norm with custom variant
        self.input_layernorm = MyRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### GatedDeltaNet (Linear Attention)

```python
GatedDeltaNet(config)
# Inputs:  hidden_states, position_embeddings (unused), past_key_value (unused)
# Outputs: output, (conv_state, recurrent_state)
```

Recurrent linear attention mechanism from the Qwen3.5 hybrid architecture
(`_gated_deltanet.py`). Key operations: fused QKV projection, causal
depthwise Conv1D, L2-normalised Q/K, exponential decay gates, delta rule
recurrence, and gated output via `GatedRMSNorm`. Supports GQA-like key
head grouping (`num_k_heads` → repeat to `num_v_heads`). State is
`conv_state` + `recurrent_state` (currently zero-initialised for stateless
export).

### RoPE variants

Created via the factory function `initialize_rope(config)`:

| `config.rope_type` | Class | Use case |
|--------------------|-------|----------|
| `"default"` | `DefaultRope` | Standard RoPE |
| `"linear"` | `LinearRope` | Linear scaling (factor in `rope_scaling`) |
| `"dynamic"` | `DynamicNTKRope` | Dynamic NTK scaling |
| `"llama3"` | `Llama3Rope` | LLaMA-3 piecewise scaling |

**MRope (Multimodal RoPE):** Two variants share a `_MRopeBase` base class
that splits frequencies into temporal (T), height (H), and width (W) sections.
`ChunkedMRope` uses a chunked layout `[TTT...HHH...WWW]` (Qwen2-VL).
`InterleavedMRope` uses an interleaved layout `[T,H,W,T,H,W,...]` and
supports `partial_rotary_factor` for partial RoPE (Qwen3-VL, Qwen3.5).

RoPE embeddings are precomputed as `cos_cache` / `sin_cache` initializers
and looked up at runtime via `Gather` on `position_ids`.

### RMSNorm

```python
RMSNorm(hidden_size, eps=1e-6)
```

Uses the ONNX `RMSNormalization` op from opset 23.  The `eps` is a float
attribute (not a Parameter).

For Gemma's `weight + 1` variant, subclass:

```python
class GemmaRMSNorm(RMSNorm):
    def forward(self, op, hidden_states):
        weight_plus_one = op.Add(self.weight, 1.0)
        return apply_rms_norm(op, hidden_states, weight_plus_one, self.variance_epsilon)
```

**OffsetRMSNorm** (`_rms_norm.py`): `output * (1 + weight)` variant where
HuggingFace stores weights initialised to 0, so the effective multiplier is
`1 + weight`. Used by Qwen3.5 for per-head Q/K normalisation.

**GatedRMSNorm** (`_rms_norm.py`): `RMSNorm(x) * SiLU(gate)` — applies
RMS normalisation then element-wise gates the result with a SiLU activation
on a separate gate input. Used by GatedDeltaNet output projection.

### LayerNorm

```python
LayerNorm(hidden_size, eps=1e-6)
```

Uses the ONNX `LayerNormalization` op.  **Always check the model's HF
config for the correct epsilon** — the default `1e-6` does not match all
models.  For example, Whisper uses `1e-5`.  A wrong epsilon causes large
numerical drift that amplifies through the network.

### LayerNormNoAffine

```python
LayerNormNoAffine(dim, eps=1e-5)
```

Layer normalization **without learnable parameters** (`elementwise_affine=False`
in PyTorch).  Used in AdaLayerNorm blocks where scale/shift come from a
separate modulation projection.  Calls `op.LayerNormalization` with no
`Scale` or `Bias` inputs.

For weight-free LayerNorm that still needs frozen ones/zeros (e.g. OLMo-1B),
create constant parameters with `data=ir.tensor(...)` instead.

**Key:** RMSNorm vs LayerNorm is NOT interchangeable.  LayerNorm subtracts
the mean; RMSNorm does not.  Using the wrong type causes max abs diff > 1.0
that grows through layers.

### GroupNorm

```python
GroupNorm(num_groups, num_channels, eps=1e-5)
```

Group normalization with learnable `weight` and `bias`.  Uses the ONNX
`GroupNormalization` op.  Commonly used in diffusion models (UNet, VAE).

### Conv2d

```python
Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1)
```

2D convolution with bias, matching `torch.nn.Conv2d(bias=True)`.  Used in
diffusion models (VAE, UNet, ControlNet) and vision patch embeddings.
Parameters: `weight` (`[out, in/groups, kH, kW]`) and `bias` (`[out]`).

### SiLU

```python
SiLU()
# SiLU (Swish) activation as a module: x * sigmoid(x)
```

Useful in `nn.Sequential` containers where an activation needs to be a
module with a `forward()` method.  For functional use, call
`get_activation("silu")` instead.

### Linear

```python
Linear(in_features, out_features, bias=False)
# Uses MatMul (+ optional Add for bias)
```

### Embedding

```python
Embedding(num_embeddings, embedding_dim, padding_idx=0)
# Uses Gather on weight matrix
```

## Design principles

1. **Favour subclasses over flags.** When a model family has a unique variant
   (e.g. Gemma's `weight + 1` norm), create a subclass rather than adding a
   boolean flag to the base class.

2. **Keep components model-agnostic.** A component should work for any model
   that has the right config fields.  Model-specific wiring belongs in the
   model module.

3. **One file per concern.** Attention in `_attention.py`, RoPE in
   `_rotary_embedding.py`, etc.  Tests co-located as `_*_test.py`.

4. **Reuse across model families.** The same `Attention` component is used by
   LLaMA, Mistral, Qwen, Phi, and others.  Only override when the
   architecture genuinely differs.

5. **Multiple reusable variants, not one-size-fits-all.** When models need
   different behaviour (e.g. MoE gates, projector types), create separate
   classes rather than cramming everything into one class with many branches.

6. **Comment generously with architecture context.** Annotate tensor shapes
   after ops (e.g. `# (N, num_heads, head_dim)`), explain multi-step
   computations (window reordering, RoPE, spatial merge), and document how
   the ONNX graph maps to the HuggingFace reference implementation.

## Common ONNX op patterns

### Scalar constants

Many ONNX ops require tensor inputs, not Python scalars:

```python
# K for TopK must be a 1-D tensor
k = op.Constant(value_ints=[2])
values, indices = op.TopK(logits, k, axis=-1)

# Integer constants
one = op.Constant(value_int=1)

# Float constants
eps = op.Constant(value_float=1e-6)
```

### Shape manipulation

Use `op.Shape` with `start` and `end` attributes to extract specific
dimensions directly — do **not** use `Gather(Shape(x), index)`:

```python
# GOOD — single Shape node with start/end
batch_size = op.Shape(x, start=0, end=1)   # 1-D [1]-element tensor
seq_len    = op.Shape(x, start=1, end=2)
hidden_dim = op.Shape(x, start=2, end=3)

# BAD — unnecessary Gather
batch_size = op.Gather(op.Shape(x), [0], axis=0)
```

Building dynamic shapes for Reshape/Concat:

```python
new_shape = op.Concat(batch_size, hidden_dim, op.Constant(value_ints=[-1]), axis=0)
reshaped = op.Reshape(x, new_shape)
```

Since `Shape(start, end)` returns a 1-D tensor, it can be passed directly
to ops expecting 1-D shape inputs (e.g. `Slice` starts/ends, `Reshape`,
`Concat` for shape building) without intermediate `Reshape` calls.

### Module lists and sequential containers

Use `nn.ModuleList` to register a list of child modules. It automatically
registers children with numeric keys (`"0"`, `"1"`, ...) and supports
iteration, indexing, and `len()`:

```python
# GOOD — nn.ModuleList
self.layers = nn.ModuleList(
    [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
)

# BAD — manual setattr loop
self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
for i, layer in enumerate(self.layers):
    setattr(self, f"layers.{i}", layer)
```

For sequential containers where children should be called in order (e.g.
matching HF `nn.Sequential`), use `nn.Sequential`. It subclasses
`nn.ModuleList` and adds automatic forward chaining:

```python
from mobius.components import Linear, SiLU

# nn.Sequential chains forward calls: SiLU → Linear
self.img_mod = nn.Sequential(SiLU(), Linear(dim, 6 * dim))

# Clean call — output chains through each child
result = self.img_mod(op, temb)  # equivalent to Linear(SiLU(temb))
```

`nn.Sequential` produces the same parameter names as `nn.ModuleList`
(`img_mod.0.weight`, `img_mod.1.weight`). The key implementation detail:
it overrides `_set_name` to keep children with simple "0", "1" names
(not fully-qualified), because `__call__` already pushes the parent name
onto the scope stack.

**When to use which:**
- `nn.Sequential` — children are called in a fixed chain (e.g. `to_out`,
  modulation layers, FFN with activation gaps)
- `nn.ModuleList` — children need custom iteration logic (e.g. decoder
  layers with residual connections, down/up blocks with skip connections)

For non-consecutive indices (e.g. matching HF `nn.Sequential` with
activation/dropout layers at skipped positions), include parameter-free
placeholder modules to fill the gaps:

```python
class _NoOpModule(nn.Module):
    """Placeholder for HF Dropout (no params, identity at inference)."""
    def forward(self, op, x):
        return x

# Matches HF net.0.proj.weight, net.2.weight (Dropout at index 1)
self.net = nn.Sequential(
    _GELUGate(dim, inner_dim * 2),  # index 0
    _NoOpModule(),                   # index 1 (Dropout placeholder)
    Linear(inner_dim, dim),          # index 2
)
result = self.net(op, x)  # chains: GELUGate → NoOp → Linear
```

If `nn.Sequential` is not available, fall back to `nn.ModuleList` with
explicit indexing:

```python
self.img_mod = nn.ModuleList([SiLU(), Linear(dim, 6 * dim)])
# Manual chaining:
result = self.img_mod[1](op, self.img_mod[0](op, temb))
```

### Conditional operations

```python
mask = op.Equal(input_ids, op.Constant(value_int=token_id))
result = op.Where(mask, true_value, false_value)
```

### Exposing parameters as graph outputs

Sometimes a generation loop needs access to model weights for external
computation (e.g. embedding lookups in numpy). Use `op.Identity()` to
expose a parameter as a graph output without affecting the initializer
name used for weight loading:

```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Stacked weight exposed for external lookup
        self.stacked_embedding = nn.Parameter([num_groups, vocab, hidden])

    def forward(self, op, ...):
        # Use Identity to create a separate output value.
        # This prevents the optimizer from renaming the initializer
        # when the task sets a custom output name.
        embeddings_out = op.Identity(self.stacked_embedding)
        return logits, present_key_values, embeddings_out
```

In the task, you can safely rename the Identity output:

```python
# Safe — Identity separates the output name from the initializer name
embeddings_out.name = "codec_embeddings"
graph.outputs.append(embeddings_out)
```

**Important:** Without the Identity node, the optimizer may fold the
reference and setting `output.name = "..."` would rename the
initializer itself, breaking `preprocess_weights` name mapping.

The generation loop extracts the weights once via a dummy inference:

```python
weights = session.run(dummy_inputs)["codec_embeddings"]  # (N, vocab, H)
# Use as numpy lookup: embed = weights[step, code_id, :]
```
