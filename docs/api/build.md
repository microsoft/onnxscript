# `build()`

Build an ONNX `ModelPackage` from a HuggingFace model ID.

```python
from mobius import build
```

## Signature

```python
def build(
    model_id: str,
    task: str | ModelTask | None = None,
    *,
    module_class: type[nn.Module] | None = None,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
) -> ModelPackage:
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | (required) | HuggingFace model repository ID (e.g. `"meta-llama/Llama-3.2-1B"`). |
| `task` | `str \| ModelTask \| None` | `None` | Model task (e.g. `"text-generation"`). Auto-detected when `None`. |
| `module_class` | `type[nn.Module] \| None` | `None` | Custom module class. Auto-detected from registry when `None`. |
| `dtype` | `str \| ir.DataType \| None` | `None` | Target dtype (`"f32"`, `"f16"`, `"bf16"`). Auto-detected from HF config when `None`. |
| `load_weights` | `bool` | `True` | Whether to download and apply weights from HuggingFace. |
| `trust_remote_code` | `bool` | `False` | Whether to trust remote code when loading the HF config. |

## Returns

`ModelPackage` — A dict-like collection of named `ir.Model` objects.

## Examples

```python
from mobius import build

# Auto-detect architecture and task
pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama/")

# Build without weights (graph only)
pkg = build("meta-llama/Llama-3.2-1B", load_weights=False)

# Override dtype
pkg = build("meta-llama/Llama-3.2-1B", dtype="f16")

# Custom module class
pkg = build("meta-llama/Llama-3.2-1B", module_class=MyCustomModule)
```

## Behavior

1. Downloads the HuggingFace config via `transformers.AutoConfig`
2. Detects `model_type` and looks up the module class in the registry
3. Falls back to diffusers pipeline detection if not a transformer model
4. Builds the ONNX graph via `build_from_module()`
5. Downloads and applies weights (if `load_weights=True`)
