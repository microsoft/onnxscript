# CLI Reference

## Usage

```bash
mobius <command> [options]
```

## `build` — Export an ONNX model

```bash
mobius build --model MODEL_ID OUTPUT_DIR [options]
```

The task is auto-detected from the model type. For example, Whisper models
automatically use `speech-to-text`, and standard LLMs use `text-generation`.

### Options

| Option | Description |
|--------|-------------|
| `--model MODEL_ID` | HuggingFace model identifier. |
| `--config PATH` | Local config directory (alternative to `--model`). |
| `--task TASK` | Model task (auto-detected if not specified). Use `list tasks` to see all available tasks. |
| `--external-data FORMAT` | External data format: `onnx` (default), `safetensors`. |
| `--max-shard-size SIZE` | Maximum shard size for safetensors (e.g. `5GB`). |
| `--no-weights` | Export graph structure only, without weight data. |
| `--dtype DTYPE` | Override model dtype: `f16`, `bf16`, `f32`. |
| `--trust-remote-code` | Trust remote code when loading the HuggingFace model config. |
| `--optimize [RULES]` | Apply rewrite rules after building. Use alone for all rules, or specify comma-separated names (e.g. `--optimize=group_query_attention,skip_norm`). |
| `--component NAME` | Build only one component from a diffusers pipeline (e.g. `--component vae_decoder`). |

### Examples

```bash
# Build from a HuggingFace model ID
mobius build --model Qwen/Qwen2.5-0.5B output_dir/

# Build without weights (graph skeleton only)
mobius build --model meta-llama/Llama-3.2-1B output_dir/ --no-weights

# Build from a local config directory
mobius build --config /path/to/model/ output_dir/

# Export with safetensors external data
mobius build --model Qwen/Qwen2.5-0.5B output_dir/ --external-data safetensors

# Build encoder-decoder model (produces encoder.onnx + decoder.onnx)
mobius build --model openai/whisper-tiny output_dir/

# Build a diffusers pipeline (auto-detected)
mobius build --model Qwen/Qwen-Image-2512 output_dir/

# Build only the VAE decoder from a diffusers pipeline
mobius build --model Qwen/Qwen-Image-2512 output_dir/ --component vae_decoder

# Apply graph optimizations
mobius build --model Qwen/Qwen2.5-0.5B output_dir/ --optimize

# Apply specific rewrite rules
mobius build --model Qwen/Qwen2.5-0.5B output_dir/ --optimize=group_query_attention,skip_norm

# Override task explicitly
mobius build --model google/gemma-3-4b-pt output_dir/ --task vision-language
```

## `list` — Discover supported models, tasks, and dtypes

```bash
mobius list {models,tasks,dtypes}
```

### Examples

```bash
# List all 130+ supported model architectures
mobius list models

# List all available tasks
mobius list tasks

# List available dtype options
mobius list dtypes
```

## `info` — Inspect a model

```bash
mobius info MODEL_ID [--trust-remote-code]
```

Shows model type, task, module class, and key config fields without building.

### Examples

```bash
# Inspect a transformers model
mobius info meta-llama/Llama-3.2-1B

# Inspect a diffusers pipeline
mobius info Qwen/Qwen-Image-2512
```
