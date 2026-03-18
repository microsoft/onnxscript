# `build_from_module()`

Build an ONNX `ModelPackage` from a module instance and config.

```python
from mobius import build_from_module
```

## Signature

```python
def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
) -> ModelPackage:
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `nn.Module` | (required) | An `onnxscript.nn.Module` instance. Its `forward()` signature must be compatible with the task. |
| `config` | `BaseModelConfig` | (required) | Architecture configuration. The `dtype` field controls precision. |
| `task` | `str \| ModelTask` | `"text-generation"` | Task name string or `ModelTask` instance. |

## Returns

`ModelPackage` — A dict-like collection of named `ir.Model` objects.

## Examples

```python
from mobius import build_from_module, ArchitectureConfig
from mobius.models import CausalLMModel

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

module = CausalLMModel(config)
pkg = build_from_module(module, config)
pkg["model"]  # ir.Model
```

## Behavior

1. Validates config (if `validate()` is available)
2. Casts module parameters to target dtype
3. Resolves the task and builds the ONNX graph
4. Applies optimization passes (identity elimination, CSE, etc.)
