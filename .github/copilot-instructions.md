# Copilot Instructions for mobius

## Build, test, and lint

```bash
# Install for development
pip install -e ".[transformers,testing]"

# Run all non-integration tests (~10 seconds)
# Add -n auto for parallel execution on multi-core machines
python -m pytest tests/build_graph_test.py tests/cli_test.py src/ -q \
  -k "not phi4mm and not apply_weights_unknown" --tb=short -n auto

# Run a single test by keyword
python -m pytest tests/build_graph_test.py -k "qwen2"

# Run integration tests (requires HuggingFace model downloads)
python -m pytest tests/integration_test.py -m integration -k "prefill" -sv

# Lint and auto-format
lintrunner f --output oneline --all-files
```

## Architecture

The package constructs ONNX models declaratively using `onnxscript.nn` —
it builds graphs directly, never tracing or exporting PyTorch.

### Four-layer stack

```
Components  →  Models  →  Tasks  →  Registry/Exporter
```

1. **Components** (`components/`): Reusable ONNX building blocks (Attention,
   MLP, RMSNorm, RoPE, etc.). Model-agnostic; shared across architectures.
2. **Models** (`models/`): Architecture-specific `nn.Module` subclasses
   (e.g. `CausalLMModel`, `GemmaCausalLMModel`). Compose components and
   define `forward()` that builds the ONNX graph, plus `preprocess_weights()`
   for HuggingFace weight name mapping.
3. **Tasks** (`tasks/`): Wire a model into an ONNX graph with specific I/O
   contracts. `CausalLMTask` creates `input_ids → logits + KV cache`.
   `VisionLanguageTask` produces 3 models (decoder, vision, embedding).
   Tasks return `ModelPackage`.
4. **Registry** (`_registry.py`): Maps HuggingFace `model_type` strings to
   `(module_class, task, config_class)` tuples. The `build(model_id)` function
   auto-detects architecture and drives the full pipeline.

### Data flow

```python
pkg = build("Qwen/Qwen2.5-VL-3B-Instruct")
# HF config → detect model_type → registry lookup → ArchitectureConfig
# → module instantiation → task.build() → ModelPackage → apply weights
# Result: pkg["model"], pkg["vision"], pkg["embedding"]
```

### ModelPackage

`build()` and `build_from_module()` return `ModelPackage` (a dict of
`str → ir.Model`). Single-model tasks produce `pkg["model"]`.
Multi-model tasks (VLM) produce `pkg["model"]`, `pkg["vision"]`,
`pkg["embedding"]`.

## Key conventions

### Module and component design

- Components accept `(op: OpBuilder, ...)` as the first forward argument.
  `op` is the ONNX op builder, not a PyTorch operator.
- Prefer subclassing over boolean flags. If a model needs different attention
  behavior, create an `XAttention(Attention)` subclass rather than adding
  flags to the base `Attention`.
- Components must be model-agnostic. Never reference a specific model name
  inside a component.
- Import components from the public API: `from mobius.components
  import Attention`, not from private submodules.

### Weight name alignment

ONNX parameter names are derived from `nn.Module` attribute names. To
minimize `preprocess_weights()` renames, align module attribute names with
HuggingFace weight name segments. Use `nn.ModuleList` for indexed layers
and wrapper modules for nesting. See the `weight-name-alignment` skill.

### File organization

- Source: `src/mobius/`
- Co-located tests: `src/mobius/**/*_test.py` (unit tests next
  to source)
- Top-level tests: `tests/build_graph_test.py` (graph construction),
  `tests/integration_test.py` (numerical accuracy vs HuggingFace)
- Skills: `.github/skills/<name>/SKILL.md`

### Code style

- `from __future__ import annotations` in every module
- Ruff for linting/formatting (line-length=95, Python 3.10+)
- MyPy strict mode (excludes `*_test.py`)
- Use `op.Shape(x, start=i, end=i+1)` for single dimension extraction
- Use ONNX opset 23 `op.Attention` with `q_num_heads`/`kv_num_heads`
  attributes (not `num_heads`)

### Comments and documentation

- **Add inline comments that explain the model architecture**: annotate
  tensor shapes, describe what each step of the forward pass is doing, and
  explain non-obvious ONNX op choices. Example:
  ```python
  # Spatial-merge permutation: (H, W) -> (H_m, ms, W_m, ms)
  # -> transpose to (H_m, W_m, ms, ms) -> flatten to (H*W,)
  h_grid = op.Reshape(h_grid, shape_4d)
  h_grid = op.Transpose(h_grid, perm=[0, 2, 1, 3])
  ```
- Annotate tensor shapes in comments after operations:
  `# (N, num_heads, head_dim)`
- Explain data flow between components, especially for multi-step
  computations like attention preprocessing, window reordering, or
  position embedding interpolation.
- Module docstrings should describe inputs, outputs, and the HuggingFace
  class being replicated.

### Adding a new model

1. Create `models/<name>.py` with model class extending `CausalLMModel` or
   appropriate base
2. Export from `models/__init__.py`
3. Register in `_registry.py`'s `_create_default_registry()`
4. Add tiny config to `_MODEL_CONFIGS` in `tests/build_graph_test.py`
5. Run tests: `python -m pytest tests/build_graph_test.py -k "<name>"`

See the `adding-a-new-model` skill for the full guide.

### Testing

- **Unit tests**: Build the ONNX graph with a tiny config (hidden=64,
  2 layers, 256 vocab). No weights, no network. Must be fast (<1s each).
- **Integration tests**: Marked `@pytest.mark.integration`. Compare
  single-forward-pass output against HuggingFace PyTorch with real weights.
  Tolerance: `atol=1e-4` for float32.
- Test files must end in `_test.py`.

### Debugging and diagnosis

- **Root-cause before remedy.** When data looks wrong, diagnose WHY it's
  wrong before prescribing a fix. Never mask symptoms by adding compensating
  logic — if two independent signals contradict each other, at least one
  signal is broken. Find which one and fix the source. Delegating a "quick
  fix" that papers over inconsistency is a quality failure.

### Git commits

Linear commit history. Always signoff commits with --signoff.
