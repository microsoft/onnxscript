# mobius

ONNX model definitions for GenAI using the `onnxscript.nn` API.

## Overview

This package provides model definitions for generative AI architectures — LLMs,
MoE, multimodal, encoder-only, encoder-decoder, vision, audio, and diffusion
models — built directly as ONNX graphs using `onnxscript.nn.Module`. Rather than
tracing or exporting PyTorch models, it **constructs** the ONNX graph
declaratively, then applies pretrained HuggingFace weights.

Supports building ONNX models from HuggingFace model IDs with automatic weight
downloading, dtype casting (including bfloat16 via `ir.LazyTensor`), and
multi-component export for pipelines.

📖 **[Documentation](https://onnxruntime.github.io/mobius/)** · 📦 **[Supported Models](https://onnxruntime.github.io/mobius/models/index.html)**

## Highlighted Models

| Category | Examples |
|---|---|
| **Text Generation** | Llama 2/3/4, Mistral, Qwen 2/2.5/3, Phi-3/3.5, Gemma 1/2/3, Granite, GPT-2, OPT, OLMo, SmolLM3, and many more |
| **Mixture of Experts** | PhiMoE, GPTOSS, Mixtral, OLMoE, DeepSeek-V2/V3, Qwen2-MoE, Qwen3-MoE, Arctic, DBRX, Jamba |
| **Multimodal** | Gemma 3, Phi-3V, Phi-4MM (vision + audio + LoRA), LLaVA, InternVL2, Qwen2.5-VL, Qwen3-VL, Pixtral |
| **Encoder-only** | BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA, XLNet |
| **Encoder-Decoder** | BART, T5/mT5, Marian, M2M-100, Pegasus, BigBird-Pegasus |
| **Speech-to-Text** | Whisper |
| **Audio** | Wav2Vec2, HuBERT, WavLM, SpeechT5 |
| **Vision** | ViT, BEiT, DeiT, DINOv2, Swin, CLIP, SigLIP |
| **Diffusion** | Stable Diffusion (UNet + VAE + ControlNet), Flux, SD3, DiT, QwenImage |
| **Adapters** | T2I-Adapter, IP-Adapter |

Supports **130 Transformers model types** and **5 Diffusers component types**
across **14 task types** and **56+ reusable components**.

See the [model documentation](https://onnxruntime.github.io/mobius/models/index.html) for the complete list.

## Installation

```bash
pip install -e .
```

For running tests:

```bash
pip install -e ".[testing]"
```

## Quick Start

### Python API

```python
from mobius import build

# Build a model package with weights
pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

### CLI

```bash
# Build from a HuggingFace model ID
mobius build --model Qwen/Qwen2.5-0.5B output_dir/

# Build without weights (graph skeleton only)
mobius build --model meta-llama/Llama-3.2-1B output_dir/ --no-weights

# Build a diffusers pipeline (all components)
mobius build --model Qwen/Qwen-Image-2512 output_dir/

# Build encoder-decoder model (produces encoder/model.onnx + decoder/model.onnx)
mobius build --model openai/whisper-tiny output_dir/

# Specify dtype
mobius build --model meta-llama/Llama-3.2-1B output_dir/ --dtype f16
```

See the [CLI reference](https://onnxruntime.github.io/mobius/cli.html) for all options.

### Examples

- [`examples/build_and_save.py`](examples/build_and_save.py) — Build and save ONNX models (simplest workflow)
- [`examples/text_generation.py`](examples/text_generation.py) — Greedy text generation with a causal LM
- [`examples/multimodal_generation.py`](examples/multimodal_generation.py) — Image captioning with a multimodal model

## Architecture

```
HuggingFace Hub
       │
       ▼
 ArchitectureConfig ◄── from_transformers() / from_diffusers()
       │
       ▼
 Model Module ◄── Reusable Components (Attention, MLP, RMSNorm, RoPE, …)
       │
       ▼
 Task ◄── CausalLMTask, VisionLanguageTask, VAETask, DenoisingTask, …
       │
       ▼
 ONNX Model ◄── preprocess_weights() + apply_weights()
```

The package is organised into four layers:

- **Components** — `onnxscript.nn.Module` building blocks (Attention, MLP,
  DecoderLayer, RoPE, VisionEncoder, MoELayer, …)
- **Models** — Full architectures composed from components
- **Tasks** — Define the ONNX graph I/O contract (inputs, outputs, KV cache)
- **Registry** — Maps HuggingFace `model_type` strings to model classes

See the [design document](https://onnxruntime.github.io/mobius/design.html) for details.

## Development

```bash
# Unit tests (fast, no network needed)
pytest tests/build_graph_test.py -v

# Integration tests (downloads models)
pytest tests/integration_test.py -m integration -v

# All unit tests (components, configs, tasks, models)
pytest src tests -m "not integration" -v

# Linting
lintrunner f --all-files
```

### Adding a new model

See the [AI-assisted model support strategy](https://onnxruntime.github.io/mobius/ai-model-support-strategy.html)
and the developer skills in `.github/skills/`:

| Skill | Use when |
|-------|----------|
| `adding-a-new-model` | Adding any new HuggingFace model architecture |
| `reusable-components` | Creating or extending components |
| `moe-models` | Adding a Mixture-of-Experts model |
| `multimodal-models` | Adding a vision-language model |
| `writing-tests` | Writing unit or integration tests |
| `writing-rewrite-rules` | Adding ONNX graph rewrite rules |
