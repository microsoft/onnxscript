# `apply_weights()`

Apply weights from a state dict to a built `ModelPackage`.

```python
from mobius import apply_weights
```

## Signature

```python
def apply_weights(
    state_dict: dict[str, torch.Tensor],
    pkg: ModelPackage,
    prefix_map: dict[str, str] | None = None,
) -> None:
```

This is also available as `ModelPackage.apply_weights()` (the preferred
interface):

```python
pkg.apply_weights(state_dict, prefix_map=prefix_map)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state_dict` | `dict[str, torch.Tensor]` | (required) | Mapping of parameter names to tensors. |
| `prefix_map` | `dict[str, str] \| None` | `None` | Mapping from weight-name prefix to component name for multi-model packages. |

## Examples

```python
import safetensors.torch
from mobius import build

# Build without weights, then apply manually
pkg = build("meta-llama/Llama-3.2-1B", load_weights=False)

state_dict = safetensors.torch.load_file("model.safetensors")
pkg.apply_weights(state_dict)
```

## Weight Routing

For multi-component packages (e.g. vision-language models), use
`prefix_map` to route weights to the correct component:

```python
pkg.apply_weights(state_dict, prefix_map={
    "model.vision": "vision",
    "model.language": "model",
})
```

Unmatched weights are tried against all components.
