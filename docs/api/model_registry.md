# `ModelRegistry`

Registry mapping HuggingFace `model_type` strings to module classes.

```python
from mobius import registry, ModelRegistry, ModelRegistration
```

## `ModelRegistry`

### `register()`

Register a module class for an architecture name.

```python
def register(
    self,
    architecture: str,
    module_class: type[nn.Module],
    *,
    task: str | None = None,
    config_class: type[BaseModelConfig] | None = None,
) -> None:
```

### `get()`

Look up the module class for an architecture.

```python
def get(self, architecture: str) -> type[nn.Module]:
```

Raises `KeyError` with suggestions if the architecture is not found.

### `architectures()`

Return a sorted list of all registered architecture names.

```python
def architectures(self) -> list[str]:
```

## `ModelRegistration`

A frozen dataclass representing a single registry entry.

```python
@dataclasses.dataclass(frozen=True)
class ModelRegistration:
    module_class: type[nn.Module]
    task: str | None = None
    config_class: type[BaseModelConfig] | None = None
```

## The Global Registry

The module-level `registry` is the default registry with all built-in
architectures pre-registered:

```python
from mobius import registry

# Check if a model is supported
"llama" in registry  # True

# Get the module class
module_class = registry.get("llama")  # CausalLMModel

# List all architectures
for arch in registry.architectures():
    print(arch)
```

## Registering Custom Models

```python
from mobius import registry

# Simple registration
registry.register("my_model", MyModelClass)

# Full registration with task and config
registry.register(
    "my_model",
    MyModelClass,
    task="text-generation",
    config_class=MyConfig,
)
```
