# GGUF Support Proposal for mobius

**Status**: Draft — awaiting go/no-go decision
**Author**: Architect (Agent 029c7dd2)
**Date**: 2026-02-27

## Executive Summary

This proposal adds the ability to **import GGUF model files** and convert
them to ONNX models using mobius' existing graph construction
pipeline. GGUF is the dominant format for local/edge inference (llama.cpp,
Ollama, LM Studio, etc.). By supporting GGUF import, we let users bring
quantized community models directly into the ONNX Runtime ecosystem without
needing to find the original HuggingFace checkpoint.

**Key insight**: We do NOT need to parse or dequantize GGUF weights
ourselves. HuggingFace Transformers already has full GGUF loading support
(`AutoModelForCausalLM.from_pretrained(repo, gguf_file=...)`) that
dequantizes GGUF tensors to fp32 PyTorch state dicts. Our value-add is
the **second path**: keeping weights quantized using either standard ONNX
**QDQ** (DequantizeLinear + MatMul) for cross-runtime portability, or
ORT's **MatMulNBits** for maximum ORT performance (see Section 8).

## 1. What is GGUF?

### 1.1 Overview

GGUF (GPT-Generated Unified Format) is a single-file binary format
created by the ggml/llama.cpp ecosystem. It stores:

- **Model metadata** as key-value pairs (architecture, hyperparameters,
  RoPE config, vocabulary, tokenizer)
- **Tensor data** with per-tensor quantization type annotations
- **Tokenizer** (vocabulary, merges, special tokens) — fully self-contained

GGUF is the successor to GGML/GGMF/GGJT formats and is the standard
for quantized model distribution on HuggingFace (>50,000 GGUF models
as of early 2026).

### 1.2 Quantization Types

GGUF supports 30+ quantization formats. The most commonly used:

| GGUF Type | Bits | Block Size | Description | Popularity |
|-----------|------|------------|-------------|------------|
| `F16` | 16 | — | IEEE float16, no quant | Common for small models |
| `BF16` | 16 | — | BFloat16 | Newer models |
| `Q8_0` | 8 | 32 | Symmetric 8-bit | High quality baseline |
| `Q4_0` | 4 | 32 | Symmetric 4-bit, simple | Fast, moderate quality |
| `Q4_1` | 4 | 32 | Asymmetric 4-bit + min | Better than Q4_0 |
| `Q5_0` | 5 | 32 | Symmetric 5-bit | Good balance |
| `Q5_1` | 5 | 32 | Asymmetric 5-bit + min | Better than Q5_0 |
| `Q2_K` | 2-4 | 256 | K-quant super-blocks, mixed | Smallest |
| `Q3_K` | 3-4 | 256 | K-quant, 3-bit dominant | Small, decent quality |
| `Q4_K` | 4-5 | 256 | K-quant, 4-bit dominant | **Most popular** |
| `Q5_K` | 5-6 | 256 | K-quant, 5-bit dominant | High quality |
| `Q6_K` | 6 | 256 | K-quant, 6-bit | Near-lossless |
| `IQ4_NL` | 4 | 32 | Non-linear 4-bit (lookup) | Newest, best q4 |
| `IQ4_XS` | 4 | 256 | Non-linear 4-bit K-quant | Newest, compact |

**K-quant** (types ending in `_K`) use a two-level "super-block"
scheme: a 256-element super-block containing 8 sub-blocks of 32
elements each, with per-super-block and per-sub-block scales.

**IQ** (importance quantization) types use non-linear quantization
with lookup tables for better quality at the same bit width.

### 1.3 Tensor Naming Convention

GGUF uses its own tensor naming, different from HuggingFace:

| GGUF Name | HuggingFace Equivalent |
|-----------|----------------------|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |
| `blk.N.attn_q.weight` | `model.layers.N.self_attn.q_proj.weight` |
| `blk.N.attn_k.weight` | `model.layers.N.self_attn.k_proj.weight` |
| `blk.N.attn_v.weight` | `model.layers.N.self_attn.v_proj.weight` |
| `blk.N.attn_output.weight` | `model.layers.N.self_attn.o_proj.weight` |
| `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` |
| `blk.N.ffn_gate.weight` | `model.layers.N.mlp.gate_proj.weight` |
| `blk.N.ffn_up.weight` | `model.layers.N.mlp.up_proj.weight` |
| `blk.N.ffn_down.weight` | `model.layers.N.mlp.down_proj.weight` |
| `blk.N.ffn_norm.weight` | `model.layers.N.post_attention_layernorm.weight` |

This mapping varies by architecture. For example, GPT-2 uses
`attn_qkv` (fused QKV), BERT uses different layer prefixes, and MoE
models use `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` for expert
weights.

The `gguf` Python package provides `get_tensor_name_map(arch, num_layers)`
which generates the complete mapping for any supported architecture.

### 1.4 Ecosystem

- **llama.cpp / ggml**: The reference runtime. Supports all GGUF types.
- **Ollama**: Popular GUI/CLI wrapper around llama.cpp. All models are GGUF.
- **LM Studio**: Desktop app for local inference. Primary format is GGUF.
- **vLLM**: Supports GGUF loading for GPU inference.
- **HuggingFace**: 50K+ GGUF models; Transformers can load GGUF → PyTorch.
- **`gguf` PyPI package**: Official Python reader/writer from llama.cpp.
  Provides `GGUFReader`, `dequantize()`, tensor name mapping utilities.

## 2. Motivation

### 2.1 User Stories

**Story 1: "I have a GGUF model, I want ORT inference"**
> A developer has a Q4_K_M quantized Llama model they use with Ollama.
> They want to deploy it with ONNX Runtime for better GPU performance
> or NPU deployment. Today they must: find the original HF checkpoint →
> download full fp16 weights → quantize with our pipeline → get ONNX.
> With GGUF import: point at the .gguf file → get ONNX directly.

**Story 2: "I want to compare llama.cpp vs ORT quality"**
> A researcher wants to benchmark the same quantized model across
> runtimes. GGUF import ensures bit-exact weight equivalence.

**Story 3: "My model only exists as GGUF"**
> Community fine-tunes are sometimes distributed only as GGUF
> (quantized by the uploader, original weights not shared). These
> models are unreachable today.

### 2.2 Why Not Just Use llama.cpp?

llama.cpp is excellent for CPU and GPU inference but:
- No execution provider ecosystem (TensorRT EP, QNN EP, OpenVINO EP, etc.)
- Limited NPU/mobile deployment (no standard deployment format)
- No graph optimization pipeline (our rewrite rules, ORT graph transformers)
- No integration with broader ONNX tooling (Olive, model zoo)

ONNX Runtime has 15+ execution providers. Converting GGUF → ONNX
unlocks all of them.

### 2.3 Competitive Landscape

| Tool | GGUF Support | Notes |
|------|-------------|-------|
| llama.cpp | Native | Reference runtime |
| vLLM | Load GGUF | GPU-focused, dequantizes to fp16 |
| HF Transformers | Load GGUF | Dequantizes to fp32, for fine-tuning |
| Optimum | ❌ | No GGUF support |
| Olive | ❌ | No GGUF support |
| **mobius** | **❌ → Proposed** | GGUF → ONNX (quantized) |

We would be the **first tool** to convert GGUF models to ONNX while
preserving quantization. This is a differentiator.

## 3. Technical Approach

### 3.1 Architecture Overview

Two import paths, sharing the same ONNX graph construction:

```
                    ┌─────────────────┐
                    │   .gguf file    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  GGUFReader     │
                    │  (gguf package) │
                    └────────┬────────┘
                             │
               ┌─────────────┼─────────────┐
               │                           │
    ┌──────────▼──────────┐   ┌────────────▼───────────┐
    │  Path A: Dequantize │   │  Path B: Keep Quantized │
    │  (fp32 state dict)  │   │  (QDQ or MatMulNBits)     │
    └──────────┬──────────┘   └────────────┬───────────┘
               │                           │
    ┌──────────▼──────────┐   ┌────────────▼───────────┐
    │  Map GGUF names     │   │  Map GGUF names         │
    │  → HF-style names   │   │  → HF-style names       │
    │                     │   │  + reshape to            │
    │                     │   │    MatMulNBits layout     │
    └──────────┬──────────┘   └────────────┬───────────┘
               │                           │
               └─────────────┬─────────────┘
                             │
                    ┌────────▼────────┐
                    │  Existing       │
                    │  build pipeline │
                    │  (model class + │
                    │   task + apply  │
                    │   weights)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ONNX model     │
                    │  (.onnx file)   │
                    └─────────────────┘
```

### 3.2 Path A: Dequantize → Standard Pipeline (Phase 1)

The simplest approach. Leverages HuggingFace's existing GGUF loading:

```python
# Conceptual flow
from gguf import GGUFReader, dequantize

reader = GGUFReader("model.gguf")

# 1. Extract config from GGUF metadata
config = gguf_metadata_to_architecture_config(reader)

# 2. Dequantize all tensors to fp32
state_dict = {}
for tensor in reader.tensors:
    name_hf = gguf_name_to_hf_name(tensor.name, config.model_type)
    weights = dequantize(tensor.data, tensor.tensor_type)
    state_dict[name_hf] = torch.from_numpy(weights)

# 3. Feed into existing pipeline
model_class, task_class, _ = registry.get(config.model_type)
module = model_class(config)
pkg = task_class.build(module, config)
pkg.apply_weights(state_dict)
```

**Pros**: Simple, uses existing infrastructure, supports all GGUF types.
**Cons**: Loses quantization (fp32 output), high memory usage.

### 3.3 Path B: Preserve Quantization → MatMulNBits (Phase 2)

The high-value path. Maps GGUF quantized tensors directly to ORT's
`MatMulNBits` operator, preserving memory efficiency:

```python
# Conceptual flow
from gguf import GGUFReader

reader = GGUFReader("model.gguf")
config = gguf_metadata_to_architecture_config(reader)

# Build graph with QuantizedLinear layers
config.quant_method = "gguf"
config.bits = 4  # detected from GGUF tensor types
config.block_size = 32  # detected from GGUF quantization scheme

module = model_class(config)  # Uses QuantizedLinear via linear_class
pkg = task_class.build(module, config)

# Reshape GGUF quantized tensors to MatMulNBits layout
state_dict = gguf_to_matmulnbits_state_dict(reader, config)
pkg.apply_weights(state_dict)
```

#### 3.3.1 Quantization Format Mapping

| GGUF Type | ORT Mapping | Approach |
|-----------|-------------|----------|
| `F32` | Standard `float32` | Direct copy |
| `F16` | Standard `float16` | Direct copy |
| `BF16` | Standard `bfloat16` | Direct copy |
| `Q8_0` | `MatMulNBits(bits=8, block_size=32)` | Repack: extract scale + int8 data |
| `Q4_0` | `MatMulNBits(bits=4, block_size=32)` | Repack: symmetric, no zero-point |
| `Q4_1` | `MatMulNBits(bits=4, block_size=32)` | Repack: asymmetric, has zero-point |
| `Q5_0` | Dequantize → `float16` | 5-bit not supported by MatMulNBits |
| `Q5_1` | Dequantize → `float16` | 5-bit not supported by MatMulNBits |
| `Q4_K` | `MatMulNBits(bits=4, block_size=32)` | Repack sub-blocks; super-block scale applied to sub-scales |
| `Q5_K` | Dequantize → `float16` | 5-bit not supported |
| `Q6_K` | Dequantize → `float16` | 6-bit not supported |
| `Q2_K` | Dequantize → `float16` | 2-bit not supported |
| `Q3_K` | Dequantize → `float16` | 3-bit not supported |
| `IQ4_NL` | Dequantize → lookup → `MatMulNBits(4,32)` | Apply lookup table then repack |
| `IQ4_XS` | Dequantize → lookup → `MatMulNBits(4,256)` | Apply lookup table then repack |

**Key constraint**: ORT's `MatMulNBits` only supports 4-bit and 8-bit
quantization. GGUF types that don't map cleanly (Q2_K, Q3_K, Q5_*,
Q6_K) must be dequantized to fp16. This is a mixed-precision model —
most layers at 4/8-bit, unsupported layers at fp16.

#### 3.3.2 Repacking Q4_0 to MatMulNBits

Q4_0 stores 32 4-bit values + 1 fp16 scale per block:

```
GGUF Q4_0 block (18 bytes):
  scale: float16          (2 bytes)
  quants: uint8[16]       (16 bytes, 32 nibbles)

MatMulNBits expects:
  weight:      [N, n_blocks, blob_size]  uint8  (blob_size = 32*4/8 = 16)
  scales:      [N, n_blocks]             float16
  zero_points: [N, ceil(n_blocks/2)]     uint8  (optional, for asymmetric)
```

The repacking extracts scales into a separate tensor, transposes the
weight matrix from `[K_packed, N]` to `[N, n_blocks, blob_size]`, and
optionally computes zero-point tensors for asymmetric types (Q4_1).

This is structurally identical to what `preprocess_gptq_weights()` and
`preprocess_awq_weights()` already do — we can share the repacking
infrastructure.

#### 3.3.3 Handling K-Quant Super-Blocks

K-quant types (Q4_K, Q5_K, etc.) use 256-element super-blocks
containing 8 sub-blocks of 32 elements. Each sub-block has its own
scale, and the super-block has a master scale + min.

For `MatMulNBits` mapping, we "flatten" the two-level scheme:
```python
effective_scale[sub] = super_scale * sub_scale
effective_zero[sub] = super_min  # applied uniformly
```

This loses some of the K-quant precision but is the closest
MatMulNBits representation. An alternative is to use
`block_size=256` and treat the super-block as a single block,
but this changes the de-quantization semantics.

**Open question**: Should we support K-quant types via dequantize-only
in Phase 2, and add native K-quant repacking in Phase 3?

### 3.4 Config Extraction

GGUF metadata maps directly to our `ArchitectureConfig`:

```python
def gguf_metadata_to_architecture_config(reader):
    """Map GGUF key-value metadata to ArchitectureConfig fields."""
    arch = read_field(reader, "general.architecture")  # e.g., "llama"

    return ArchitectureConfig(
        model_type=arch,
        vocab_size=read_field(reader, f"{arch}.vocab_size"),
        hidden_size=read_field(reader, f"{arch}.embedding_length"),
        num_hidden_layers=read_field(reader, f"{arch}.block_count"),
        intermediate_size=read_field(reader, f"{arch}.feed_forward_length"),
        num_attention_heads=read_field(reader, f"{arch}.attention.head_count"),
        num_key_value_heads=read_field(reader, f"{arch}.attention.head_count_kv"),
        rms_norm_eps=read_field(reader, f"{arch}.attention.layer_norm_rms_epsilon"),
        rope_theta=read_field(reader, f"{arch}.rope.freq_base"),
        max_position_embeddings=read_field(reader, f"{arch}.context_length"),
        # ... additional fields per architecture
    )
```

HuggingFace's `GGUF_CONFIG_MAPPING` (in `integrations/ggml.py`) already
defines these mappings for 15+ architectures. We can reference or adapt
this mapping.

### 3.5 Tensor Name Mapping

We need to convert GGUF tensor names to HuggingFace-style names so they
work with our existing `preprocess_weights()` pipeline.

**Approach A — Use the `gguf` package's `get_tensor_name_map()`**:

The `gguf` package (from llama.cpp) provides bidirectional tensor name
mapping. We can use it to convert GGUF names to HF names, then feed
through our standard weight loading pipeline. This is what HuggingFace
Transformers does internally.

```python
from gguf import get_tensor_name_map, MODEL_ARCH_NAMES

arch_key = next(k for k, v in MODEL_ARCH_NAMES.items() if v == model_type)
name_map = get_tensor_name_map(arch_key, num_layers)

# name_map.get_name("blk.0.attn_q") → "model.layers.0.self_attn.q_proj"
```

**Approach B — Build our own mapping table**:

Smaller, no dependency on `gguf` package internals. But more maintenance.

**Recommendation**: Approach A. The `gguf` package is well-maintained
(same repo as llama.cpp) and the mapping is authoritative.

### 3.6 Architecture-Specific Tensor Processing

Some architectures need tensor transformations during GGUF loading
(beyond name mapping):

| Architecture | Transformation | Reason |
|---|---|---|
| Llama/Mistral | Reverse-permute Q/K weights | llama.cpp permutes Q/K for RoPE |
| GPT-2 | Transpose Linear weights | Conv1D → Linear layout |
| Bloom | Reshape fused QKV | Different head interleaving |
| Mamba | Expand conv1d dims, exp(A) | Different storage convention |
| Nemotron/Gemma2 | Subtract 1 from norm weights | Different norm convention |

HuggingFace's `TENSOR_PROCESSORS` (in `modeling_gguf_pytorch_utils.py`)
implements all of these. We can reference their logic or delegate to
their code if the user has transformers installed.

## 4. Proposed Architecture

### 4.1 Module Location

```
src/mobius/
├── integrations/
│   └── gguf/
│       ├── __init__.py          # Public API
│       ├── _reader.py           # GGUF file reading + metadata extraction
│       ├── _config_mapping.py   # GGUF metadata → ArchitectureConfig
│       ├── _tensor_mapping.py   # GGUF tensor names → HF-style names
│       ├── _repacker.py         # GGUF quant blocks → MatMulNBits layout
│       ├── _tensor_processor.py # Architecture-specific transforms
│       └── _reader_test.py      # Unit tests (co-located)
```

### 4.2 Public API

```python
# Phase 1: Dequantize path
from mobius.integrations.gguf import build_from_gguf

pkg = build_from_gguf("path/to/model.gguf")
# Returns ModelPackage with fp32/fp16 ONNX model

# Phase 2: Quantized path
pkg = build_from_gguf("path/to/model.gguf", keep_quantized=True)
# Returns ModelPackage with MatMulNBits ONNX model
```

### 4.3 CLI Integration

```bash
# Phase 1
mobius build --gguf path/to/model.gguf --output model.onnx

# Phase 2 (preserve quantization)
mobius build --gguf path/to/model.gguf --keep-quantized \
    --output model.onnx
```

### 4.4 Integration with Existing Pipeline

The GGUF integration plugs into the existing 4-layer stack:

1. **GGUFReader** extracts metadata → creates `ArchitectureConfig`
2. **Registry** looks up model class + task by `model_type`
3. **Task** builds ONNX graph (standard or quantized)
4. **Weight loading** receives HF-style state dict from GGUF mapper

This reuses 100% of the existing model classes, tasks, and graph
construction. The only new code is the GGUF → HF bridge.

## 5. Dependencies

### 5.1 Required

- **`gguf>=0.10.0`** (PyPI): Official GGUF reader/writer from llama.cpp.
  Provides `GGUFReader`, `dequantize()`, tensor name mapping.
  ~50KB pure Python package, no native dependencies.

### 5.2 Optional

- **`transformers`** (already optional dep): For tokenizer extraction
  from GGUF. GGUF embeds the tokenizer, but constructing a proper
  `tokenizer.json` requires HF's converter classes.

### 5.3 Dependency Strategy

Add `gguf` as an optional dependency in a new extras group:

```toml
[project.optional-dependencies]
gguf = ["gguf>=0.10.0"]
# Combined:
all = ["mobius-ai[transformers,gguf]"]
```

The `gguf` import is lazy — users who don't use GGUF features
never import it. The `build_from_gguf()` function raises
`ImportError` with install instructions if `gguf` is missing.

## 6. Scope and Phasing

### Phase 1: Dequantized Import (MVP)

**Goal**: Load any GGUF file → construct fp32 ONNX model.

- Parse GGUF metadata → `ArchitectureConfig`
- Map GGUF tensor names → HF-style names
- Dequantize all tensors via `gguf.dequantize()`
- Run architecture-specific tensor processors (Q/K permute, etc.)
- Feed state dict through existing `preprocess_weights()` + `apply_weights()`
- Support architectures: Llama, Mistral, Qwen2, Gemma2, Phi3, Falcon,
  GPT-2, Mamba (matching HF's GGUF support matrix)

**Effort**: M (Medium) — ~500-800 lines of new code
**Dependencies**: `gguf` package only
**Risk**: Low — HuggingFace has proven this path works

### Phase 2: Quantized Import (High Value)

**Goal**: Load Q4_0/Q4_1/Q8_0 GGUF → quantized ONNX with MatMulNBits.

- Detect quantization type from GGUF tensor metadata
- Repack Q4_0/Q4_1/Q8_0 blocks → MatMulNBits `[N, n_blocks, blob_size]`
- Build ONNX graph with `QuantizedLinear` layers
- Mixed-precision: quantized Linear weights + fp16/fp32 norms/embeddings
- K-quant types (Q4_K): flatten super-block scales to per-block scales

**Effort**: L (Large) — ~1000-1500 lines, careful bit manipulation
**Dependencies**: Phase 1 complete
**Risk**: Medium — repacking logic needs extensive numerical validation.
K-quant super-block flattening may lose precision.

### Phase 3: Extended Quantization + Export

**Goal**: Support all GGUF types + ONNX → GGUF export.

- IQ types (importance quantization): lookup table dequantization
- Q2_K/Q3_K/Q5_K/Q6_K: dequantize to fp16 (no MatMulNBits mapping)
- ONNX → GGUF export: implement GGUF writer using `gguf` package
- Round-trip validation: GGUF → ONNX → GGUF produces equivalent model

**Effort**: XL (Extra Large)
**Dependencies**: Phase 2 complete
**Risk**: High — GGUF export requires writing a complex binary format
and handling edge cases across architectures. May not be worth the
effort if the primary use case is import.

## 7. Quantization Format Deep Dive

### 7.1 Q4_0 (Simple Symmetric 4-bit)

Block layout (18 bytes per 32 elements):
```
struct block_q4_0 {
    ggml_half d;       // scale (float16, 2 bytes)
    uint8_t qs[16];    // 32 x 4-bit values packed into 16 bytes
};
```

Dequantization: `x[i] = (qs[i] - 8) * d`

The values are unsigned 0-15, recentered to -8..+7 via subtracting 8
(implicit zero-point = 8).

**MatMulNBits mapping**: Direct. Scale → `scales` tensor. Packed nibbles
→ `weight` tensor. Zero-point = 8 → `zero_points` tensor.

### 7.2 Q4_1 (Asymmetric 4-bit)

Block layout (20 bytes per 32 elements):
```
struct block_q4_1 {
    ggml_half d;       // scale (float16)
    ggml_half m;       // minimum (float16)
    uint8_t qs[16];    // 32 x 4-bit values packed into 16 bytes
};
```

Dequantization: `x[i] = qs[i] * d + m`

**MatMulNBits mapping**: scale → `scales`, min → derived zero-point.
`zero_point = round(-m / d)`. This is the same asymmetric pattern as
AWQ/GPTQ.

### 7.3 Q4_K (K-Quant 4-bit)

Super-block layout (144 bytes per 256 elements):
```
struct block_q4_K {
    ggml_half d;           // super-block scale
    ggml_half dmin;        // super-block minimum
    uint8_t scales[12];    // 8 sub-block scales + mins, 6-bit packed
    uint8_t qs[128];       // 256 x 4-bit values
};
```

Each sub-block (32 elements) has its own 6-bit scale and min, but
these are further scaled by the super-block `d` and `dmin`.

Dequantization:
```
sub_scale = decode_6bit(scales, sub_idx) * d
sub_min = decode_6bit(scales, sub_idx + 8) * dmin
x[i] = qs[i] * sub_scale - sub_min
```

**MatMulNBits mapping**: Flatten to `block_size=32`:
```
effective_scale[sub] = decode_6bit(...) * d
effective_zero[sub] = round(decode_6bit(...) * dmin / effective_scale[sub])
```

This is a lossy approximation — the two-level scale hierarchy doesn't
map perfectly to MatMulNBits' single-level scale+zero_point scheme.
Quality impact needs benchmarking.

### 7.4 Q8_0 (Symmetric 8-bit)

Block layout (34 bytes per 32 elements):
```
struct block_q8_0 {
    ggml_half d;       // scale
    int8_t qs[32];     // 32 x int8 values
};
```

**MatMulNBits mapping**: Direct with `bits=8, block_size=32`. Simplest
repacking — just separate scale from data and transpose.

## 8. ONNX Quantization Representations: QDQ vs MatMulNBits

The GGUF proposal's Path B ("keep quantized") must choose an ONNX
representation for quantized weights. There are two candidates: the
standard **QDQ pattern** (QuantizeLinear/DequantizeLinear) and ORT's
proprietary **MatMulNBits** contrib op. This section analyzes both in
depth and recommends a dual-path strategy.

### 8.1 QDQ Representation (Standard ONNX)

The QDQ ("Quantize-DeQuantize") pattern uses standard ONNX operators to
represent weight-only quantization. For inference, the pattern is
weight-only — `QuantizeLinear` is used during model preparation but only
`DequantizeLinear` appears in the inference graph:

```
[quantized_weight: int4/uint4, shape (N, K)]  ← initializer
[scale: float16, shape (N, ceil(K/B))]        ← initializer
[zero_point: int4/uint4, shape (N, ceil(K/B))]← initializer (optional)
           │        │           │
           ▼        ▼           ▼
    ┌──────────────────────────┐
    │   DequantizeLinear       │
    │   axis=1, block_size=B   │
    └────────────┬─────────────┘
                 │  (float16, shape N×K)
                 ▼
    ┌──────────────────────────┐
    │   MatMul(input, weight^T)│
    └──────────────────────────┘
```

**Key attributes** (as of ONNX opset 21+):
- **`axis`**: Which dimension to quantize along (typically `1` for the
  reduction dimension K in a weight matrix).
- **`block_size`**: Number of elements sharing the same scale/zero-point.
  When `block_size=32`, every 32 elements along `axis` share a scale.
  This is the ONNX equivalent of group/block quantization.

**Data type support** (opset evolution):
| ONNX Version | Opset | int4/uint4 | block_size | Notes |
|---|---|---|---|---|
| 1.14 | 19 | ❌ | ❌ | int8/uint8 only, per-axis only |
| 1.15 | 20 | ❌ | ❌ | Added float8 types |
| 1.16 | 21 | ✅ | ✅ | **First int4 + block quantization** |
| 1.17 | 22 | ✅ | ✅ | Stable |
| 1.18 | 23 | ✅ | ✅ | Added output_dtype control |
| 1.19 | 24 | ✅ | ✅ | Added float8e8m0 scale type |
| 1.20 | 25 | ✅ | ✅ | Added int2/uint2, float8e8m0 scales |

**int4 packing format** (from ONNX spec): Two 4-bit values per byte.
First element in 4 LSB, second in 4 MSB. For odd tensor sizes, 4 bits
of padding are appended. Storage size = `ceil(N/2)` bytes.

```python
# ONNX int4 packing
pack(x, y) = (y << 4) | (x & 0x0F)
unpack(z)  = x = z & 0x0F,  y = z >> 4
```

**Dequantization formula**: `y = (x - zero_point) * scale`

For blocked quantization with `block_size=B` along `axis=1`:
- Weight shape: `(N, K)` stored as int4 → `(N, ceil(K/2))` bytes
- Scale shape: `(N, ceil(K/B))` as float16
- Zero-point shape: `(N, ceil(K/B))` as int4 (optional; defaults to 0)

### 8.2 MatMulNBits Representation (ORT Contrib Op)

`com.microsoft.MatMulNBits` is a fused operator in ORT's contrib domain
that performs dequantization and matrix multiplication in a single kernel:

```
[packed_weight: uint8, shape (N, n_blocks, blob_size)]  ← initializer
[scales: float16, shape (N, n_blocks)]                   ← initializer
[zero_points: uint8, shape (N, ceil(n_blocks/2))]        ← optional
           │        │           │
           ▼        ▼           ▼
    ┌──────────────────────────┐
    │   MatMulNBits            │
    │   K, N, bits, block_size │
    │   domain=com.microsoft   │
    └────────────┬─────────────┘
        input ──►│
                 ▼
            [output: float16, shape (*, N)]
```

**Attributes**:
- **`K`**: Inner dimension (reduction dim) of the weight matrix.
- **`N`**: Output dimension of the weight matrix.
- **`bits`**: Quantization bit-width (4 or 8 only).
- **`block_size`**: Elements per quantization group (power of 2, ≥16).
- **`accuracy_level`**: Optional, tunes dequant precision vs speed.

**Inputs**: `(A, B, scales, zero_points, g_idx, bias)` where `A` is
the fp16/fp32 activation, `B` is the packed uint8 weight blob.

**Weight packing**: N-bit values are packed into uint8 blobs:
- `blob_size = block_size * bits / 8`
- `n_blocks = ceil(K / block_size)`
- For 4-bit: two values per byte, `blob_size = block_size / 2`
- For 8-bit: one value per byte, `blob_size = block_size`
- Zero-points (4-bit): packed two per byte → `ceil(n_blocks / 2)` bytes

### 8.3 Head-to-Head Comparison

| Dimension | QDQ (DequantizeLinear + MatMul) | MatMulNBits |
|---|---|---|
| **Standard** | ✅ ONNX standard ops (opset 21+) | ❌ `com.microsoft` contrib only |
| **Portability** | ✅ Any ONNX runtime | ❌ ORT only |
| **TensorRT EP** | ✅ Fuses DQ+MatMul into INT4 kernel | ❌ Not recognized |
| **OpenVINO EP** | ✅ Supported (plugin-dependent) | ❌ Not recognized |
| **QNN EP** | ✅ Parses QDQ patterns for NPU | ❌ Not recognized |
| **CUDA EP (ORT)** | ✅ DQ+MatMul fusion available | ✅ Native fused kernel |
| **CPU EP (ORT)** | ⚠️ Limited INT4 fusion | ✅ Optimized VNNI/AVX |
| **Bit-widths** | 2/4/8-bit + float4/float8 | 4/8-bit only |
| **Graph nodes** | 2 nodes per Linear (DQ + MatMul) | 1 node per Linear |
| **Model size** | Same (int4 packed + scales) | Same (uint8 blob + scales) |
| **Kernel efficiency** | Depends on EP fusion quality | Single fused kernel |
| **Model validation** | `onnx.checker` validates | Requires ORT-specific check |

### 8.4 Detailed Pros/Cons

#### QDQ Pros

1. **Universal portability**: Standard ONNX ops work with any compliant
   runtime — ORT, TensorRT, OpenVINO, QNN, XNNPACK, CoreML, etc. This
   is the single strongest argument for QDQ.

2. **EP fusion ecosystem**: All major EPs have invested in recognizing
   `DequantizeLinear → MatMul` patterns and fusing them into optimized
   kernels. TensorRT fuses to its INT4 GEMM kernels. OpenVINO maps to
   its quantized inference pipeline. QNN maps to Snapdragon NPU
   quantized ops. This is the standard optimization path.

3. **Future-proof**: As ONNX adds new quantization types in future
   opsets, QDQ automatically supports them. MatMulNBits requires
   explicit ORT changes for each new type.

4. **Tooling support**: ORT's quantization toolkit, NVIDIA Model
   Optimizer, AMD Vitis AI quantizer, and Intel Neural Compressor all
   produce QDQ-format models. Broad tooling interop.

5. **Explicit semantics**: Scale and zero-point are separate, typed
   tensors with clear mathematical meaning. The dequantization formula
   `y = (x - zp) * scale` is unambiguous.

6. **Block quantization**: Opset 21+ `block_size` attribute maps
   directly to GGUF's per-32-element quantization blocks (Q4_0, Q8_0).

7. **QAT compatibility**: QDQ is the standard format for
   Quantization-Aware Training (QAT) models from NVIDIA Model
   Optimizer, Intel Neural Compressor, etc. This means QDQ provides
   a unified representation for both PTQ and QAT workflows.

#### QDQ Cons

1. **Uniform quantization only**: QDQ assumes linear mapping
   `y = (x - zp) * scale`. GGUF's non-linear IQ types (IQ4_NL, IQ4_XS)
   use lookup tables that have no QDQ equivalent. These must be
   dequantized or the lookup applied before QDQ packing.

2. **No nested/hierarchical blocking**: GGUF K-quant types (Q4_K, Q5_K)
   use 256-element super-blocks containing 8 sub-blocks of 32, with
   two-level scale hierarchies (super-scale × sub-scale). QDQ's
   `block_size` is single-level only. Representing Q4_K requires
   flattening: `effective_scale = super_scale × sub_scale`, which is
   lossy because the zero-point relationship
   `effective_zp = super_min / effective_scale` introduces rounding.

3. **Two graph nodes per Linear**: `DequantizeLinear` + `MatMul` vs
   MatMulNBits' single node. More nodes increase graph complexity and
   rely on EP fusion to achieve equivalent performance. If an EP fails
   to fuse (e.g., unusual shape, unsupported config), performance
   degrades to "dequantize then fp matmul" — 2-4× slower.

4. **INT4 requires opset 21+**: Our codebase uses opset 23, so this
   isn't a blocker, but older runtimes (pre-2024) can't load int4 QDQ
   models.

5. **Packing format mismatch**: ONNX int4 packs LSB-first (first
   element in low nibble), while GGUF Q4_0 packs differently (unsigned
   0-15 values). Repacking is needed regardless of representation.

#### MatMulNBits Pros

1. **Single fused kernel**: One node = one kernel dispatch. No fusion
   required. Guaranteed performance regardless of EP optimization level.

2. **ORT-optimized**: Highly tuned CUDA kernels (with accuracy_level
   control), AVX2/AVX512/VNNI CPU kernels. ORT's own quantization
   pipeline (GPTQ, AWQ, RTN) all target MatMulNBits.

3. **Direct GGUF mapping**: Q4_0/Q4_1/Q8_0 block structure maps
   directly to MatMulNBits' `(N, n_blocks, blob_size)` layout with
   minimal repacking (extract scale, transpose, repack nibbles).

4. **Proven in production**: mobius already uses
   `QuantizedLinear` (our existing component) which emits MatMulNBits.
   GPTQ/AWQ weight loading is built around this representation.

5. **Packed zero-points**: 4-bit zero-points are packed two per byte,
   matching GGUF's compact storage. QDQ uses full int4 tensors (same
   packing, but conceptually less explicit about the packing).

#### MatMulNBits Cons

1. **ORT-only**: The critical weakness. Models using MatMulNBits cannot
   run on TensorRT, OpenVINO, QNN, CoreML, or any non-ORT runtime.
   This directly conflicts with the GGUF proposal's motivation: "ONNX
   Runtime has 15+ execution providers — converting GGUF → ONNX unlocks
   all of them."

2. **No standard validation**: `onnx.checker.check_model()` does not
   validate `com.microsoft` domain ops. Model validity depends on
   runtime-specific checks.

3. **Limited bit-widths**: Only 4 and 8-bit. No 2-bit, 3-bit, 5-bit,
   6-bit. This means Q2_K, Q3_K, Q5_K, Q6_K GGUF types must all be
   dequantized even in the "keep quantized" path.

4. **No standard evolution path**: If ONNX standardizes a native fused
   quantized matmul op, MatMulNBits models won't benefit automatically.
   Migration would require graph rewriting.

### 8.5 GGUF Type → ONNX Representation Mapping

| GGUF Type | QDQ Mapping | MatMulNBits Mapping | Recommended |
|---|---|---|---|
| `F32`/`F16`/`BF16` | N/A (use as-is) | N/A (use as-is) | Standard fp ops |
| **`Q4_0`** | DQ(int4, scale, block=32) sym | MMNB(bits=4, block=32) sym | **Both work cleanly** |
| **`Q4_1`** | DQ(uint4, scale+zp, block=32) asym | MMNB(bits=4, block=32) + zp | **Both work cleanly** |
| **`Q8_0`** | DQ(int8, scale, block=32) sym | MMNB(bits=8, block=32) sym | **Both work cleanly** |
| `Q5_0`/`Q5_1` | DQ(int8, ...) with 5→8 padding | ❌ Must dequantize | QDQ with int8 or dequantize |
| `Q4_K` | DQ(int4, flattened_scale, block=32) | MMNB(4,32) + flattened scale | Both lossy, QDQ preferred |
| `Q5_K` | DQ(int8, flattened_scale, block=32) | ❌ Must dequantize | QDQ with int8 or dequantize |
| `Q6_K` | DQ(int8, flattened_scale, block=32) | ❌ Must dequantize | QDQ with int8 or dequantize |
| `Q2_K` | DQ(int4, ...) lossy 2→4 promotion | ❌ Must dequantize | Dequantize to fp16 |
| `Q3_K` | DQ(int4, ...) lossy 3→4 promotion | ❌ Must dequantize | Dequantize to fp16 |
| `IQ4_NL` | ❌ Non-linear, no QDQ mapping | ❌ Non-linear | **Dequantize only** |
| `IQ4_XS` | ❌ Non-linear, no QDQ mapping | ❌ Non-linear | **Dequantize only** |

**Key insight**: Q4_0, Q4_1, and Q8_0 are the only GGUF types that map
cleanly to _both_ representations. These are also the simplest and most
common non-K-quant types. Q4_K (the most popular K-quant) requires
lossy super-block flattening regardless of representation.

### 8.6 Limitations of Both Representations

Neither QDQ nor MatMulNBits can natively represent:

1. **Hierarchical/nested block quantization**: GGUF K-quant's two-level
   super-block structure (256 elements → 8 sub-blocks of 32, with
   per-super-block and per-sub-block scales) has no ONNX equivalent.
   Both representations flatten to single-level `block_size=32`.

2. **Non-uniform/lookup-table quantization**: GGUF IQ types use learned
   codebooks or non-linear mappings. QDQ is strictly linear (`y = (x - zp) * scale`).
   MatMulNBits is strictly linear. No standard ONNX op represents
   codebook-based vector quantization.

3. **Mixed-precision within a tensor**: Q4_K uses fp16 super-scales +
   6-bit sub-scales + 4-bit weights — three precision levels in one
   tensor. Both representations support only one scale type per tensor.

4. **Importance matrix (imatrix) metadata**: GGUF can embed the
   importance matrix used during quantization. Neither ONNX
   representation preserves this metadata (it's not needed for inference
   but is useful for requantization).

5. **Odd bit-widths** (5-bit, 6-bit): QDQ can approximate via
   promotion to int8 (wasting storage), but there's no native 5-bit or
   6-bit ONNX type. MatMulNBits doesn't support these at all.

### 8.7 Recommendation: Dual-Path Strategy

We recommend **QDQ as the primary representation** with **MatMulNBits
as an ORT-specific optimization option**:

```python
# Default: portable QDQ model (works everywhere)
pkg = build_from_gguf("model.gguf", keep_quantized=True)

# ORT-optimized: MatMulNBits model (best ORT performance)
pkg = build_from_gguf("model.gguf", keep_quantized=True,
                       quant_format="matmulnbits")
```

**Rationale**:

1. **QDQ first** because portability is the primary value proposition
   of ONNX. If a user converts GGUF → ONNX but can only run on ORT,
   we've reduced the value of the conversion (they could use llama.cpp
   directly). QDQ unlocks TensorRT, OpenVINO, QNN — the real
   differentiator.

2. **MatMulNBits as opt-in** for users who know they're targeting ORT
   and want guaranteed fused-kernel performance without relying on EP
   fusion. This is our existing `QuantizedLinear` component — zero new
   code for the graph construction side.

3. **Rewrite rule** (future): A `QDQ_to_MatMulNBits` rewrite rule could
   convert QDQ models to MatMulNBits for ORT deployment, decoupling the
   "how we represent" question from "how we export." This is consistent
   with our rewrite rule architecture.

**Implementation impact on Path B phasing**:

| Phase | Path B: Keep Quantized |
|---|---|
| Phase 2a | QDQ for Q4_0/Q4_1/Q8_0 (clean mapping, portable) |
| Phase 2b | MatMulNBits alternative via `quant_format=` flag |
| Phase 2c | QDQ for Q4_K (flattened super-blocks, lossy but usable) |
| Phase 3 | Q5_K/Q6_K via QDQ int8 promotion (storage-inefficient) |

### 8.8 New Component: QDQLinear

To support the QDQ path, we need a `QDQLinear` component alongside the
existing `QuantizedLinear` (MatMulNBits):

```python
class QDQLinear(nn.Module):
    """Linear layer using standard DequantizeLinear + MatMul pattern.

    Portable across all ONNX runtimes. EPs fuse DQ+MatMul into
    optimized quantized kernels.
    """

    def __init__(self, in_features, out_features, bits=4, block_size=32,
                 has_zero_point=False, bias=False):
        super().__init__()
        n_blocks = math.ceil(in_features / block_size)
        q_dtype = ir.DataType.INT4 if bits == 4 else ir.DataType.INT8

        # Quantized weight stored as int4/int8
        self.weight = nn.Parameter(
            [out_features, in_features], dtype=q_dtype
        )
        # Per-block scales: (N, n_blocks) for blocked quantization
        self.scales = nn.Parameter(
            [out_features, n_blocks], dtype=ir.DataType.FLOAT16
        )
        self.zero_points = (
            nn.Parameter([out_features, n_blocks], dtype=q_dtype)
            if has_zero_point else None
        )
        self._block_size = block_size
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op, x):
        inputs = [self.weight, self.scales]
        if self.zero_points is not None:
            inputs.append(self.zero_points)
        # DequantizeLinear: int4 weight → float16
        dq_weight = op.DequantizeLinear(
            *inputs, axis=1, block_size=self._block_size
        )
        # Standard MatMul with transposed dequantized weight
        result = op.MatMul(x, op.Transpose(dq_weight, perm=[1, 0]))
        if self.bias is not None:
            result = op.Add(result, self.bias)
        return result
```

> **⚠️ EP Fusion Note:** The `Transpose` between `DequantizeLinear` and
> `MatMul` may break execution provider fusion patterns. TensorRT and
> other EPs recognize `DQ → MatMul` as a fused quantized matmul —
> inserting a Transpose may prevent fusion. Two mitigations:
> 1. Store weights as `(K, N)` with `axis=0` quantization → direct
>    `MatMul(x, dq_weight)` with no Transpose
> 2. Use a rewrite rule to fold the Transpose before EP compilation
>
> Benchmark both layouts before committing to a storage convention.

This parallels `QuantizedLinear` but produces portable ONNX. The
`linear_class` pattern in our model architecture already supports
injecting either component:

```python
# QDQ (portable)
linear_class = make_qdq_linear_factory(bits=4, block_size=32)

# MatMulNBits (ORT-optimized)
linear_class = make_quantized_linear_factory(bits=4, block_size=32)
```

### 8.9 Proposed ONNX Feature Request: Hierarchical Block Quantization

Neither QDQ nor MatMulNBits handles K-quant super-blocks well. We
should propose an ONNX spec extension:

---

**Title**: Support hierarchical/nested block quantization in
DequantizeLinear

**Motivation**: Emerging quantization formats (GGUF K-quant, HQQ) use
multi-level scale hierarchies: a coarse "super-block" of 256 elements
contains 8 fine "sub-blocks" of 32 elements, each with its own scale.
The super-block has a master scale that modulates the sub-block scales.
Current `DequantizeLinear` supports only single-level `block_size`.

**Current behavior**: `block_size=B` applies one scale per B elements.
To represent a 256-element super-block with 32-element sub-blocks, users
must flatten: `effective_scale[i] = super_scale × sub_scale[i]`. This is
lossy because:
1. The zero-point relationship involves division:
   `effective_zp = round(super_min / effective_scale)`, introducing
   rounding error.
2. The flattened scales consume more storage (one fp16 per 32 elements
   instead of one fp16 per 256 + 8 × 6-bit per 256).
3. The mathematical equivalence only holds approximately for asymmetric
   types.

**Proposed extension** (two options):

*Option A — Multi-level scales*: Allow `x_scale` to have more
dimensions than `x`, representing a scale hierarchy. For a 2-level
scheme: `x_scale` shape `(N, n_super_blocks, n_sub_blocks)` with
`block_size=[256, 32]` as a list.

*Option B — Nested DequantizeLinear*: Allow `x_scale` itself to be the
output of another `DequantizeLinear`, creating a scale-of-scales chain.
The inner DQ dequantizes 6-bit sub-scales using the super-block scale.

**Use cases**:
- GGUF Q4_K/Q5_K/Q6_K import into ONNX without precision loss
- HQQ (Half-Quadratic Quantization) which also uses multi-level scales
- Future quantization research exploring hierarchical schemes

**Backward compatibility**: The current `block_size=int` semantic is
unchanged. The extension adds `block_size=list[int]` as an optional
variant.

---

### 8.10 Summary Decision Matrix

| Scenario | Recommended Representation |
|---|---|
| GGUF Q4_0/Q8_0 → portable ONNX | **QDQ** (DequantizeLinear + MatMul) |
| GGUF Q4_0/Q8_0 → ORT-only deployment | **MatMulNBits** (maximum perf) |
| GGUF Q4_K → any runtime | **QDQ** with flattened scales (lossy) |
| GGUF Q5_*/Q6_K → any runtime | **Dequantize to fp16** (no clean quantized mapping) |
| GGUF IQ4_* → any runtime | **Dequantize to fp16** (non-linear) |
| GPTQ/AWQ models (existing) | **MatMulNBits** (status quo, works well) |
| New quantization methods | **QDQ first**, MatMulNBits rewrite rule |

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| K-quant repacking precision loss | Medium | Benchmark perplexity: GGUF-direct vs GGUF→ONNX. Accept if <0.5 PPL increase. |
| `gguf` package API instability | Low | Pin to `>=0.10.0`, use only stable APIs (`GGUFReader`, `dequantize`). |
| Architecture coverage gaps | Low | Start with Llama (90% of GGUF models). Add others incrementally. |
| QDQ INT4 EP fusion gaps | Medium | Benchmark QDQ vs MatMulNBits per-EP; provide `quant_format` flag for user choice. |
| Tensor permutation bugs | Medium | Validate against HF's GGUF loading (they've battle-tested these transforms). |

### 9.2 Product Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Low adoption if users prefer llama.cpp | Medium | Position as "GGUF → ORT for deployment" not "replace llama.cpp". |
| Maintenance burden of GGUF format evolution | Low | GGUF format is stable (v3). New quant types can be dequantized as fallback. |
| User confusion about quality differences | Medium | Clear CLI output: "Q4_K → MatMulNBits(4,32): approximate repacking, quality may differ slightly from llama.cpp". |

## 10. Alternatives Considered

### 10.1 Use HuggingFace's GGUF Loading Directly

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(repo, gguf_file=file)
state_dict = model.state_dict()  # fp32
# Then use our existing pipeline
```

**Pros**: Zero new code for reading GGUF.
**Cons**: Always dequantizes to fp32 (no quantized path). Requires
full `transformers` installation. Creates an unnecessary PyTorch model
in memory just to extract weights.

**Verdict**: This could be a quick-start option for Phase 0, but
doesn't achieve the quantized import goal.

### 10.2 Convert GGUF → Safetensors First

Use `llama.cpp/convert_hf_to_gguf.py` in reverse, or dequantize GGUF
to safetensors, then use our standard HF pipeline.

**Pros**: Completely avoids GGUF code in our codebase.
**Cons**: Loses quantization. Requires an extra conversion step.
User experience is poor.

**Verdict**: Not viable — defeats the purpose.

### 10.3 Implement GGUF Reader from Scratch

Parse GGUF binary format ourselves without the `gguf` package.

**Pros**: No new dependency.
**Cons**: 2000+ lines to reimplement a well-maintained library.
Maintenance burden for format changes.

**Verdict**: Not worth it. The `gguf` package is 50KB, pure Python,
Apache-2.0 licensed, and actively maintained.

## 11. Success Criteria

### Phase 1

- [ ] `build_from_gguf("llama-3-8b.Q4_K_M.gguf")` produces valid ONNX model
- [ ] Output model runs correctly under ORT inference
- [ ] Logits within `atol=1e-4` of HF's GGUF→PyTorch→forward()
- [ ] Supports Llama, Mistral, Qwen2, Phi3 architectures
- [ ] CLI: `mobius build --gguf model.gguf` works

### Phase 2

- [ ] Q4_0/Q4_1/Q8_0 produce MatMulNBits ONNX models
- [ ] Model file size within 10% of original GGUF (not inflated)
- [ ] ORT inference speed competitive with fp16 model (MatMulNBits should be faster)
- [ ] Perplexity within 0.5 PPL of llama.cpp on same GGUF file

## 12. Open Questions

1. **Should we vendor the `gguf` package?** It's small (~50KB), but
   vendoring avoids version conflicts. HuggingFace chose NOT to vendor
   (they require `gguf>=0.10.0`). Recommend: don't vendor.

2. **How to handle GGUF-only architectures?** Some GGUF models use
   architecture names that don't exist in our registry (e.g., `command-r`,
   `internlm2`). We need a mapping from GGUF architecture names to our
   registry's `model_type` values.

3. **Tokenizer extraction**: GGUF embeds the tokenizer. Should we
   extract and convert it to `tokenizer.json` format, or require the
   user to provide a tokenizer separately? HF Transformers extracts the
   tokenizer. We should do the same.

4. **K-quant precision**: The super-block → single-block flattening
   for Q4_K is lossy. How much quality do we lose? This needs
   benchmarking before committing to Phase 2.

5. **5-bit and 6-bit support**: MatMulNBits doesn't support 5/6-bit.
   Could we propose a MatMulNBits extension to ORT? Or are these niche
   enough that dequantize → fp16 is acceptable?

## 13. Recommendation

**Proceed with Phase 1 (Dequantize Import) as a P1 item in Sprint 9.**

Phase 1 is medium effort, low risk, and delivers immediate value.
It validates the GGUF→ONNX pipeline end-to-end and gives us user
feedback before investing in the more complex quantized path.

Phase 2 (Quantized Import) should be a P0 item in Sprint 10-11,
contingent on Phase 1 user validation and K-quant precision benchmarks.

Phase 3 (Export) should be deferred indefinitely unless there is
strong user demand. The primary value is import, not round-tripping.
