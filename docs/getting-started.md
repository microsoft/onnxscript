# Getting Started

**mobius** generates ONNX models for common GenAI architectures
(LLMs, MoE, multimodal, speech-to-text, encoder-only, encoder-decoder, vision,
audio, and diffusion) built directly as ONNX graphs using
`onnxscript.nn.Module`. It supports building from HuggingFace model IDs with
automatic weight downloading and conversion, including bfloat16 models via
`ir.LazyTensor`-based dtype casting.

📦 [GitHub Repository](https://github.com/microsoft/mobius)

## Installation

```bash
pip install mobius-ai
```

For development and testing:

```bash
pip install -e ".[testing]"
```

### Dependencies

- **Required**: `onnxscript`, `onnx_ir`, `numpy`, `torch`, `huggingface_hub`, `tqdm`
- **Optional (transformers)**: `transformers`, `safetensors` — for building from HuggingFace model IDs
- **Optional (gguf)**: `gguf` — for building from GGUF files
- **Optional (testing)**: `onnxruntime-easy`, `transformers`, `safetensors` — for running tests

## Quick Start

### Build from a HuggingFace model ID

```python
from mobius import build

pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

### Build from a custom module

```python
from mobius import build_from_module
from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel

config = ArchitectureConfig(
    vocab_size=32000,
    max_position_embeddings=4096,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    head_dim=128,
    pad_token_id=0,
)

model = build_from_module(CausalLMModel(config), config)
model["model"]  # ir.Model
```

### Build a multi-component model

Models with multiple components (e.g. encoder + decoder, or vision + text)
can be exported as a package:

```python
from mobius import build

pkg = build("openai/whisper-tiny")
pkg.save("/output/whisper/")
# Produces encoder.onnx and decoder.onnx
```

### Build different model types

```python
from mobius import build

# Encoder-only (BERT)
pkg = build("bert-base-uncased")

# Encoder-decoder (T5)
pkg = build("google-t5/t5-small")

# Multimodal (LLaVA)
pkg = build("llava-hf/llava-1.5-7b-hf")

# Audio encoder (Wav2Vec2)
pkg = build("facebook/wav2vec2-base")
```

### Build from a GGUF file

Convert a GGUF model (e.g. from llama.cpp) to ONNX:

```python
from mobius.integrations.gguf import build_from_gguf

pkg = build_from_gguf("path/to/model.gguf")
pkg.save("output/model/")
```

Or via CLI:

```bash
mobius build-gguf path/to/model.gguf --output output/model/
```

> **Note**: GGUF support requires the optional `gguf` package:
> `pip install mobius-ai[gguf]`

### Build quantized models (GPTQ/AWQ)

Quantized HuggingFace models are handled automatically:

```python
from mobius import build

# GPTQ quantized
pkg = build("TheBloke/Llama-2-7B-GPTQ")

# AWQ quantized
pkg = build("TheBloke/Llama-2-7B-AWQ")
```

### Choose a target dtype

```python
from mobius import build

# Build with float16 weights
pkg = build("meta-llama/Llama-3.2-1B", dtype="f16")

# Build with bfloat16 weights
pkg = build("meta-llama/Llama-3.2-1B", dtype="bf16")
```

### Extensibility

Register custom architectures or tasks:

```python
from mobius import registry
from mobius.tasks import TASK_REGISTRY

registry.register("my_arch", MyCustomModel)
TASK_REGISTRY["my-task"] = MyCustomTask
```

### Load config from a local directory

```python
from mobius import ArchitectureConfig

config = ArchitectureConfig.from_file("/path/to/model/")
config.validate()  # Check field consistency
```

### Apply graph optimizations

```python
from onnxscript.rewriter import rewrite
from mobius import build
from mobius.rewrite_rules import group_query_attention_rules, skip_norm_rules

pkg = build("Qwen/Qwen2.5-0.5B")
model = pkg["model"]
rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
rewrite(model, pattern_rewrite_rules=skip_norm_rules())
```

Or via CLI:

```bash
mobius build --model Qwen/Qwen2.5-0.5B output/ --optimize
```

## CLI Reference

The package provides the `mobius` CLI with these subcommands:

### `build` — Build an ONNX model from HuggingFace

```bash
# From a HuggingFace model ID
mobius build --model meta-llama/Llama-3.2-1B output/

# From a local config directory (with safetensors weights)
mobius build --config /path/to/model/ output/

# With options
mobius build --model Qwen/Qwen2.5-0.5B output/ \
    --dtype f16 \
    --external-data safetensors \
    --optimize
```

Key flags:
- `--dtype` — Target dtype (`f32`, `f16`, `bf16`)
- `--no-weights` — Build graph only (no weight download)
- `--external-data` — `onnx` (default) or `safetensors`
- `--optimize` — Apply rewrite rules (e.g. fused attention)
- `--component` — Build only one component from a diffusers pipeline

### `build-gguf` — Build from a GGUF file

```bash
mobius build-gguf model.gguf --output output/
```

### `list` — List supported resources

```bash
mobius list models   # All supported architectures
mobius list tasks    # Available task types
mobius list dtypes   # Supported dtypes
```

### `info` — Inspect a model

```bash
mobius info meta-llama/Llama-3.2-1B
```

### Adding a new model

See the [adding-a-new-model skill](../.github/skills/adding-a-new-model/SKILL.md)
for copy-paste examples and step-by-step instructions.

## Output Format

### ModelPackage

`build()` and `build_from_module()` return a `ModelPackage` — a dict-like
collection of named `ir.Model` objects.

- **Single-model tasks** (e.g. CausalLM): `pkg["model"]`
- **Multi-model tasks** (e.g. VLM): `pkg["model"]`, `pkg["vision"]`, `pkg["embedding"]`
- **Encoder-decoder**: `pkg["encoder"]`, `pkg["decoder"]`

### Saving to disk

```python
pkg.save("output/")
# Single model: output/model.onnx + output/model.onnx.data
# Multi model: output/model/model.onnx, output/vision/model.onnx, ...
```

Safetensors format:
```python
pkg.save("output/", external_data="safetensors")
```

### Discover supported models

```bash
# List all supported architectures
mobius list models

# Inspect a specific model
mobius info meta-llama/Llama-3.2-1B
```

## Development

### Running tests

```bash
# Unit tests (fast, no network needed)
pytest tests/build_graph_test.py -v

# Run a single model type
pytest tests/build_graph_test.py -k "phi4mm"

# Integration tests (downloads models, requires more time/memory)
pytest tests/integration_test.py -m integration -v

# Run a single integration test model
pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"
```

### Linting

```bash
lintrunner f --all-files
```
