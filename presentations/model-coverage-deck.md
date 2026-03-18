---
theme: default
title: "mobius: Model Coverage & Project Overview"
info: |
  Leadership overview of the mobius project —
  declarative ONNX model construction for onnxruntime-genai.
author: mobius team
keywords: onnx, genai, model coverage
exportFilename: model-coverage-deck
drawings:
  persist: false
transition: slide-left
mdc: true
---

# mobius

## Model Coverage & Project Overview

<br/>

**March 2026**

<style>
h1 { font-size: 2.5em !important; }
</style>

---
layout: two-cols
---

# What is mobius?

A Python library that **constructs ONNX models declaratively** for onnxruntime-genai — no PyTorch tracing, no export.

<br/>

### Why declarative?

- 🎯 **Deterministic** — same input → same graph
- 🚫 **No runtime deps** — no need to run PyTorch model
- 🔧 **Full control** — every ONNX op placed intentionally
- ⚡ **Optimizable** — rewrite rules at build time

::right::

<br/><br/>

```python
from mobius import build

pkg = build("Qwen/Qwen2.5-0.5B")
# → Optimized ONNX model with HF weights
#   ready for onnxruntime-genai inference
```

<br/>

| | |
|---|---|
| **Input** | HuggingFace model ID |
| **Output** | ONNX model(s) for ORT GenAI |
| **Method** | Direct graph construction via `onnxscript.nn` |

---
layout: section
---

# Coverage at a Glance

---

# Coverage Dashboard

| Metric | Count |
|--------|------:|
| Registered model types | **273** |
| Unit tests (graph construction) | **505** |
| Integration tests (real HF weights) | **109** |
| Model definition files | **61** |
| Reusable components | **32** |
| Task implementations | **22** |
| Architecture families | **15+** |

---

# Architecture Categories

| Category | Count | Examples |
|----------|:-----:|---------|
| **Causal LM** (decoder-only) | 110 | Llama, Qwen2/3, Gemma 2/3, Phi-3, Granite, Mistral, GPT-2, Mixtral, DeepSeek |
| **Encoder** | 41 | BERT, RoBERTa, DeBERTa, ALBERT, ModernBERT, XLNet |
| **Vision-Language** | 39 | Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, LLaVA, InternVL2, Gemma3, Phi-4-MM |
| **Vision** | 27 | ViT, CLIP, DINOv2, BEiT, Swin, SAM2, Hiera, MobileViT |
| **Encoder-Decoder** | 23 | BART, T5/mT5/UL2, Marian, mBART, Pegasus, Switch Transformers |
| **Audio** | 16 | Wav2Vec2, HuBERT, WavLM, Wav2Vec2-BERT |
| **Other** (multimodal, SSM, hybrid) | 12 | Phi-4-MM, Bamba, Jamba, Mamba, Qwen3.5 text/VL/MoE |
| **Speech** | 4 | Whisper, Qwen3-ASR, Qwen3-TTS |
| **Detection** | 1 | YOLOS |

Causal LM includes MoE variants (Mixtral, DeepSeek-V2/V3, OLMoE) and hybrid SSM+Attention (Jamba, Bamba).

---
layout: section
---

# Confidence Levels

---

# Confidence Levels (L0–L5)

| Level | What It Proves | Models | % |
|-------|----------------|-------:|--:|
| 🟣 **L5: Generation verified** | End-to-end generation matches golden reference | **0*** | 0% |
| 🔵 **L4: Golden match** | Whole models: Forward-pass output matches golden files | **1*** | <1% |
| 🟢 **L3: Synthetic parity** | Tiny models: Output matches HuggingFace PyTorch | **115** | 42% |
| 🟡 **L2: Config compatible** | Architecture validated with real HF config | **48** | 18% |
| 🟠 **L1: Graph builds** | ONNX graph builds with tiny synthetic params | **65** | 24% |
| ⚪ **L0: Not tested** | Registered but no test coverage yet | **44** | 16% |

<br/>

> L4-L5 test cases to be generated

---

# L1/L2 — Graph Build & Config Validation

505 unit tests covering model types with tiny synthetic parameters (hidden=64, 2 layers, vocab=256).

**Validates:** graph structure, op correctness, shape inference, KV cache plumbing.

L2 models additionally pass architecture validation with real HuggingFace configs.

---
layout: two-cols
---

# L3+ — Integration Tested Models

Real HuggingFace weights, numerical parity verified (atol ≤ 1e-4).

<br/>

**Text Generation**
- Qwen2.5-0.5B, SmolLM-135M
- Gemma-3-1B, Granite-3.3-2B
- Phi-3.5-mini, Qwen3-0.6B
- OLMo-1B, GPT-2

**MoE**
- Phi-tiny-MoE, GraniteMoE-1B
- OLMoE-1B, Qwen1.5-MoE-2.7B

::right::

<br/><br/><br/>

**Vision-Language**
- Qwen2.5-VL-3B (text + full + 3-model)
- Qwen3-VL-2B

**Encoder** — BERT, DistilBERT, RoBERTa, ALBERT

**Encoder-Decoder** — BART, T5-small, Marian

**Vision** — ViT, DINOv2, BEiT

**Audio** — Wav2Vec2, HuBERT

**Speech** — Whisper-tiny


---
layout: section
---

# How It Works

---
layout: two-cols
---

# The 4-Layer Stack

```
┌──────────────┐
│  Components  │  32 reusable ONNX blocks
│  Attention, MLP, RoPE, MoE ...
├──────────────┤
│    Models    │  61 architecture classes
│  forward() + preprocess_weights()
├──────────────┤
│     Tasks    │  22 I/O contracts
│  CausalLM, VL, Speech, Diffusion
├──────────────┤
│   Registry   │  273 model_type mappings
│  build("model-id") → auto-detect
└──────────────┘
```

::right::

<br/>

# 5-Step New Model Process

<br/>

1. **Config** — Extract architecture params
2. **Model class** — Compose components
3. **Register** — Add model_type mapping
4. **Weight map** — HF → ONNX alignment
5. **Test** — Unit test + integration test

<br/>

---
layout: section
---

# Recent Wins

---

# Qwen 3.5 — Hybrid Linear Attention ⭐

First model with **DeltaNet + standard attention hybrid** architecture:

- Linear attention layers
- **Causal convolution** for local context in linear attention layers
- Implemented as **ONNX function ops** — portable reference implementations
- Proposed as official ONNX ops ([onnx/onnx#7689](https://github.com/onnx/onnx/issues/7689))

<br/>

### Other Key Wins

| Win | Impact |
|-----|--------|
| **Test Infrastructure** | L0–L5 pyramid, 20+ golden files, AST-based impact analysis |

---

# Component Reuse — The Multiplier

32 reusable components enable rapid model onboarding:

| Component | Shared By |
|-----------|-----------|
| `Attention` (GQA/MQA/MHA) | All decoder-only LLMs |
| `RoPE` / `DynamicNTKRope` / `LongRoPE` | Llama, Qwen, Phi, Gemma, ... |
| `MoELayer` + `TopKGate` | Mixtral, PhiMoE, Qwen-MoE, OLMoE, ... |
| `MambaBlock` / `Mamba2Block` | Mamba, Jamba, Bamba, Falcon-H1 |
| `BertEncoder` | 41 encoder models |
| `ViTEncoder` | 27 vision models |
| `Wav2Vec2Encoder` | All audio models |
| `QuantizedLinear` | GPTQ and AWQ quantized variants |
