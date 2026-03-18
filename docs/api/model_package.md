# `ModelPackage`

A dict-like collection of named `ir.Model` objects forming a complete model.

```python
from mobius import ModelPackage
```

## Class Signature

```python
class ModelPackage(UserDict[str, ir.Model]):
    config: object | None

    def __init__(
        self,
        models: dict[str, ir.Model] | None = None,
        config: object | None = None,
    ) -> None: ...
```

## Methods

### `save()`

Save all component models to a directory.

```python
def save(
    self,
    directory: str,
    *,
    external_data: str = "onnx",
    max_shard_size_bytes: int | None = None,
    components: Callable[[str], bool] | None = None,
    progress_bar: bool = True,
    check_weights: bool = True,
) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `directory` | `str` | (required) | Output directory path. |
| `external_data` | `str` | `"onnx"` | `"onnx"` or `"safetensors"` format. |
| `max_shard_size_bytes` | `int \| None` | `None` | Max shard size (safetensors only). |
| `components` | `Callable \| None` | `None` | Predicate to select components to save. |
| `check_weights` | `bool` | `True` | Verify all initializers have weight data. |

### `load()`

Load models from a directory.

```python
@classmethod
def load(cls, directory: str) -> ModelPackage:
```

### `apply_weights()`

Apply weights from a state dict to all component models.

```python
def apply_weights(
    self,
    state_dict: dict[str, torch.Tensor],
    prefix_map: dict[str, str] | None = None,
) -> None:
```

## Examples

```python
from mobius import build

# Build and save
pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama/")

# Access individual models
model = pkg["model"]
print(model.graph.name)

# Check components
print(list(pkg.keys()))  # ["model"] for single-model
# ["model", "vision", "embedding"] for VLM

# Load from disk
pkg = ModelPackage.load("output/llama/")

# Save as safetensors
pkg.save("output/llama/", external_data="safetensors")
```

## Output Layout

- **Single model**: `directory/model.onnx` + `directory/model.onnx.data`
- **Multi model**: `directory/{name}/model.onnx` for each component
