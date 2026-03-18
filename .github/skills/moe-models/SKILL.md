---
name: moe-models
description: >
  How to represent Mixture-of-Experts models in mobius. Covers gate
  variants (TopKGate, SparseMixerGate), MoELayer composition, expert weight
  naming conventions, and preprocess_weights mappings. Use this skill when
  adding or modifying a model that uses MoE layers.
---

# Skill: Mixture-of-Experts (MoE) Models

## When to use

Use this skill when adding or modifying a model that uses Mixture-of-Experts
layers — where each token is routed to a subset of expert MLPs.

## Architecture overview

```
MoEDecoderLayer
 ├── RMSNorm (input_layernorm)
 ├── Attention (self_attn)
 ├── RMSNorm (post_attention_layernorm)
 └── MoELayer
      ├── Gate (routing: input → expert selection + weights)
      └── Experts[0..N-1] (each is a standard MLP)
```

### Key components

| Component | File | Purpose |
|-----------|------|---------|
| `MoELayer` | `components/_moe.py` | Routes tokens to experts, combines outputs |
| `TopKGate` | `components/_moe.py` | Standard softmax + top-k routing |
| `SparseMixerGate` | `components/_moe.py` | Sequential selection with threshold masking |
| `MoEDecoderLayer` | `models/moe.py` | Decoder layer that uses `MoELayer` instead of `MLP` |
| `MoETextModel` | `models/moe.py` | Text model with MoE decoder layers |

## How routing gates work

### TopKGate (default)

Standard routing used by most MoE models (Mixtral, GPTOSS):

1. Compute router logits: `logits = MatMul(hidden_states, gate_weight)`
2. Top-k selection: `values, indices = TopK(logits, k=num_experts_per_tok)`
3. Softmax over selected experts: `weights = Softmax(values)`

### SparseMixerGate (PhiMoE)

Sequential expert selection with threshold-based masking:

1. Compute router logits via `MatMul`
2. For each of `top_k` rounds:
   - Find max score (`ReduceMax`)
   - Threshold mask: experts whose scores are far from the max (relative to
     `jitter_eps`) are masked with `-inf`
   - Softmax over non-masked experts
   - `TopK(k=1)` to select best expert
   - `ScatterElements` to mask out the selected expert for the next round
3. Concatenate all selected expert indices and weights

## Adding a new MoE model

### 1. Determine the gate type

Check the HuggingFace implementation for the routing logic.  Look for:

- `router_type` or `routing_type` in the config
- How `router_logits` are computed and processed
- Whether top-k is applied before or after softmax

If neither `TopKGate` nor `SparseMixerGate` fits, create a new gate class
in `components/_moe.py`.

### 2. Check expert MLP naming

HuggingFace MoE models often use different weight names for expert MLPs:

| HF name | Our name | Description |
|---------|----------|-------------|
| `w1` | `gate_proj.weight` | Gate projection |
| `w2` | `down_proj.weight` | Down projection |
| `w3` | `up_proj.weight` | Up projection |

Implement a `_rename_moe_expert_weights()` function if the naming differs:

```python
def _rename_moe_expert_weights(state_dict):
    renamed = {}
    for key, value in state_dict.items():
        new_key = key
        if ".experts." in key:
            new_key = new_key.replace(".w1.", ".gate_proj.")
            new_key = new_key.replace(".w2.", ".down_proj.")
            new_key = new_key.replace(".w3.", ".up_proj.")
        renamed[new_key] = value
    return renamed
```

### 3. Check the normalization

Some MoE models use different norms than standard models:

- **PhiMoE**: Uses `LayerNorm` (with bias), not `RMSNorm`
- **Mixtral**: Uses standard `RMSNorm`

### 4. Create the model class

Use `MoETextModel` with the correct gate factory:

```python
class MyMoECausalLMModel(CausalLMModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        # Pass gate_factory for custom routing
        self.model = MoETextModel(config, gate_factory=SparseMixerGate)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def preprocess_weights(self, state_dict):
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)
```

### 5. Inject a custom gate

`MoELayer` accepts an optional `gate` parameter.  `MoETextModel` accepts
`gate_factory` — a callable `(config) -> gate_instance` that creates a gate
per layer:

```python
# Default (TopKGate)
MoETextModel(config)

# Custom gate
MoETextModel(config, gate_factory=SparseMixerGate)
```

To create a new gate, implement a class with this interface:

```python
class MyGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])
        # ... other params

    def forward(self, op, hidden_states):
        # Returns: (expert_weights, expert_indices)
        # expert_weights: [batch, seq, num_experts_per_tok]
        # expert_indices: [batch, seq, num_experts_per_tok] (INT64)
        ...
        return weights, indices
```

## Config fields for MoE

```python
config.num_local_experts     # Total number of experts (e.g. 8, 16)
config.num_experts_per_tok   # Experts activated per token (e.g. 2)
```

These are extracted automatically from HuggingFace configs.

## TopK ONNX op gotcha

The ONNX `TopK` op requires `K` as a **1-D int64 tensor**, not a Python int:

```python
# WRONG
op.TopK(logits, self.top_k, axis=-1)

# CORRECT
k_tensor = op.Constant(value_ints=[self.top_k])
op.TopK(logits, k_tensor, axis=-1)
```

## Qwen3.5-MoE: Hybrid Attention + MoE

Qwen3.5-MoE combines the hybrid DeltaNet/full-attention architecture of
Qwen3.5 dense with MoE FFN layers instead of a dense MLP.

### Architecture

```
Qwen35MoEDecoderLayer
 ├── OffsetRMSNorm (input_layernorm)
 ├── GatedDeltaNet / Qwen35Attention (per layer_types)
 ├── OffsetRMSNorm (post_attention_layernorm)
 └── Qwen35MoEBlock
      ├── TopKGate (router)
      ├── Experts[0..N-1] (standard MLP: gate/up/down with SiLU)
      ├── SharedExpert (MLP: gate/up/down with SiLU)
      └── shared_expert_gate → sigmoid scalar
```

### Classes

| Class | File | Purpose |
|-------|------|---------|
| `Qwen35MoEBlock` | `models/qwen.py` | MoE block with routed + shared experts |
| `Qwen35MoEDecoderLayer` | `models/qwen.py` | Hybrid attention + MoE FFN layer |
| `Qwen35MoETextModel` | `models/qwen.py` | Stacks decoder layers with RoPE |
| `Qwen35MoECausalLMModel` | `models/qwen.py` | Top-level causal LM model |

### MoE block (`Qwen35MoEBlock`)

- **TopKGate routing**: 256 experts, top-8 in the full model (configurable
  via `num_local_experts` / `num_experts_per_tok`)
- **Expert MLPs**: Standard `MLP` (gate/up/down projections, SiLU activation),
  each with `moe_intermediate_size` as the intermediate dim
- **Shared expert**: A separate `MLP` that runs on **all** tokens (not routed),
  sized by `shared_expert_intermediate_size`
- **Shared expert gating**: `sigmoid(shared_expert_gate(x)) * shared_expert(x)`,
  where `shared_expert_gate` is `Linear(hidden_size, 1, bias=False)`

The key difference from standard MoE is the shared expert: its output is
gated by a learned sigmoid scalar and added to the routed expert output.

### Config fields

```python
config.moe_intermediate_size           # Intermediate size per expert MLP
config.shared_expert_intermediate_size  # Intermediate size for the shared expert
config.num_local_experts               # Total number of routed experts (e.g. 256)
config.num_experts_per_tok             # Experts activated per token (e.g. 8)
config.layer_types                     # Per-layer attention type list
```

### Weight naming

HuggingFace weights map directly (no renames needed for expert names):

```
mlp.gate.weight                               → router logits
mlp.experts.N.{gate,up,down}_proj.weight      → per-expert MLP
mlp.shared_expert.{gate,up,down}_proj.weight  → shared expert MLP
mlp.shared_expert_gate.weight                 → sigmoid gate (Linear, no bias)
```

Note: HF checkpoints store experts as fused tensors
(`experts.gate_up_proj`, `experts.down_proj`).  `preprocess_weights()`
unpacks these into per-expert tensors and also renames
`linear_attn.conv1d.weight` → `linear_attn.conv1d_weight`.

### Testing

Integration tests use a random-weight HF model with reduced layers and
experts (e.g. 4 layers, 4 experts, top-2 routing).  See
`test_qwen35_moe_prefill_logits_match` in `tests/integration_test.py`.

## Testing MoE models

MoE integration tests require a model with MoE layers.  A good test model
should be small enough for CI (~1-4B params).  The test pattern:

1. Build ONNX model with weights
2. Run prefill + decode against HuggingFace reference
3. Optionally test greedy generation (token-ID matching)

See `tests/moe_integration_test.py` for the complete pattern.
