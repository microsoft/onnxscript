# `build_from_gguf()`

Build an ONNX `ModelPackage` from a GGUF model file.

```python
from mobius.integrations.gguf import build_from_gguf
```

> **Note**: Requires the optional `gguf` package:
> `pip install mobius-ai[gguf]`

## Signature

```python
def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = False,
) -> ModelPackage:
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gguf_path` | `str \| Path` | (required) | Path to a `.gguf` model file. |
| `task` | `str \| None` | `None` | Override the model task (e.g. `"text-generation"`). When `None`, the task is auto-detected from the model type. |
| `dtype` | `str \| None` | `None` | Override model dtype (e.g. `"f16"`). When `None`, defaults to float32. |
| `keep_quantized` | `bool` | `False` | When `True`, preserve quantization for supported GGUF types (Q4_0, Q4_1, Q8_0) by repacking linear-layer weights into MatMulNBits format. Unsupported types are dequantized to float. |

## Returns

`ModelPackage` — A dict-like collection of named `ir.Model` objects.

## Examples

```python
from mobius.integrations.gguf import build_from_gguf

# Basic conversion (dequantizes to float)
pkg = build_from_gguf("llama-3.2-1b-q4_0.gguf")
pkg.save("output/llama/")
```

```python
# Preserve quantization (Q4_0/Q4_1/Q8_0 → MatMulNBits)
pkg = build_from_gguf("llama-3.2-1b-q4_0.gguf", keep_quantized=True)
pkg.save("output/llama-q4/")
```

```bash
# Via CLI
mobius build-gguf llama-3.2-1b-q4_0.gguf --output output/llama/
```

## Behavior

1. Reads GGUF metadata to detect architecture and config
2. Maps GGUF tensor names to HuggingFace weight names
3. Dequantizes quantized tensors to float (or repacks into MatMulNBits
   when `keep_quantized=True`)
4. Applies architecture-specific tensor processors (e.g. Q/K permute)
5. Builds the ONNX graph using the same pipeline as `build()`
6. Runs `preprocess_weights()` (HF → ONNX name mapping)
7. Applies weights to the graph

## Supported GGUF Architectures

The GGUF builder maps GGUF architecture names (e.g. `llama`, `qwen2`,
`gemma`) to the same model classes used by `build()`. Most decoder-only
LLM architectures are supported.
