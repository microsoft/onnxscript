---
name: writing-tests
description: >
  Patterns and conventions for writing tests in mobius. Covers unit
  tests (graph construction), integration tests (numerical parity with
  HuggingFace), generation tests, testing utilities, and tolerance guidelines.
  Use this skill when adding tests for new or modified models and components.
---

# Skill: Writing Tests

## When to use

Use this skill whenever you add a new model, component, or modify existing
behaviour.  The project has a five-level confidence system (L1–L5) plus
shared configuration infrastructure.

## Confidence levels (L1–L5)

Each level is detected and counted **independently**.  A model can pass L3
(integration test) without passing L2 (no YAML test case defined), or have
L4 golden data without passing L3.  Levels are not hierarchical in detection,
though logically higher levels usually imply lower ones.

| Level | Name | What it verifies | Data source |
|-------|------|-----------------|-------------|
| **L1** | Graph builds | ONNX graph builds from a tiny synthetic config | `_MODEL_CONFIGS` / `_SPECIALIZED_TEST_MODEL_TYPES` in `tests/build_graph_test.py` |
| **L2** | Config compatible | Full-size HuggingFace config produces a valid graph | `test_model_id` field in YAML test case (`testdata/cases/`) |
| **L3** | Synthetic parity | Random-weight forward pass matches HuggingFace numerically | `tests/integration_test.py` parametrized tests |
| **L4** | Golden match | Real-weight prefill logits match pre-computed golden reference | `*.json` files in `testdata/golden/` |
| **L5** | Generation verified | Full multi-token generation matches golden output | `*_generation.json` files in `testdata/golden/` |

### How counts work on the dashboard

The dashboard shows **per-flag counts**: L1 count = models with `l1_graph_build=True`,
L2 count = models with `l2_arch_validation=True`, etc.  A model is counted at
every level it passes — not just the highest one.  Because all registered model
types have at least one graph build test, L1 equals the total number of
registered models.

## Test architecture overview

```
tests/
├── build_graph_test.py       # L1: graph construction (no weights)
├── _test_configs.py          # shared model configs for all tests
├── integration_test.py       # L3: real-weight numerical parity
├── e2e_golden_test.py        # L4 + L5: golden file comparison
├── yaml_schema_test.py       # YAML test case schema validation
└── arch_validation_test.py   # L2: full HF config graph build

testdata/
├── cases/                    # YAML test case definitions (L2, L4, L5)
│   ├── causal-lm/
│   ├── vision-language/
│   ├── audio/
│   └── ...
└── golden/                   # Pre-computed reference outputs
    ├── causal-lm/
    │   ├── gpt2.json              # L4 prefill logits
    │   └── gpt2_generation.json   # L5 generation tokens
    └── ...
```

### Running tests

```bash
# All non-integration tests (fast, no downloads)
python -m pytest tests/build_graph_test.py tests/cli_test.py src/ -q \
  -k "not phi4mm and not apply_weights_unknown" --tb=short

# Representative models only (~5 seconds)
python -m pytest tests/build_graph_test.py --fast

# Single model type
python -m pytest tests/build_graph_test.py -k "phi4mm"

# Integration tests (slow, downloads models)
python -m pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"

# L4/L5 golden tests
python -m pytest tests/e2e_golden_test.py -m golden --level L4 -v
python -m pytest tests/e2e_golden_test.py -m golden --level L5 -v
```

---

## Shared test configuration (`tests/_test_configs.py`)

All model configs for parametrized tests live in `tests/_test_configs.py`,
organized by category:

| List | Test class | Task type |
|------|-----------|-----------|
| `CAUSAL_LM_CONFIGS` | `TestBuildGraph` | text-generation |
| `ENCODER_CONFIGS` | `TestBuildEncoderGraph` | feature-extraction |
| `SEQ2SEQ_CONFIGS` | `TestBuildSeq2SeqGraph` | seq2seq |
| `VISION_CONFIGS` | `TestBuildVisionGraph` | image-classification |
| `DETECTION_CONFIGS` | `TestBuildDetectionGraph` | object-detection |

Each entry is a 3-tuple: `(model_type, config_overrides, is_representative)`.

### The `is_representative` flag

Set `True` for models with **unique behaviour** — custom model class,
softcapping, parallel attention, ALiBi, MoE routing, partial rotary,
non-standard activation, etc.  Set `False` for models that are simple
aliases of a base class with no special config.

Representative models are always tested.  Non-representative models are
skipped when running with `--fast`.

### The `--fast` flag

`pytest --fast` skips non-representative parametrized tests, reducing
run time to ~5 seconds.  Non-parametrized tests (VLM, Whisper, TTS, etc.)
always run regardless of this flag.

The flag is implemented in `tests/conftest.py` via
`pytest_collection_modifyitems`.

### Auto-generation from registry

Model types registered with `text-generation` or `hybrid-text-generation`
tasks that have **no explicit entry** in `_test_configs.py` get an
auto-generated `(model_type, {}, False)` entry.  This ensures new
registrations get basic graph-build coverage without editing test files.

Auto-generation only covers text-generation models because other tasks
(vision-language, speech, diffusion) require specialised config overrides
that cannot be guessed.

### Adding a config for a new model

```python
# In tests/_test_configs.py, add to the appropriate list:
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    # ...
    ("my_model", {"hidden_act": "gelu", "attn_qkv_bias": True}, True),
]
```

Then run:
```bash
python -m pytest tests/build_graph_test.py -k "my_model"
```

---

## L1: Graph build tests

Located in `tests/build_graph_test.py`. Uses tiny synthetic
`ArchitectureConfig` objects (64 hidden, 2 layers, 256 vocab) to test
every model type without downloading weights or requiring network access.
Configs are defined in `tests/_test_configs.py` (see above).

VLM, audio, and other specialised models have dedicated test methods
(not parametrized via `_test_configs.py`) and are tracked in
`_SPECIALIZED_TEST_MODEL_TYPES`.  The dashboard L1 scanner covers both.

### Rewrite rule unit tests

Rewrite rule unit tests should be placed **next to** the source file they
test, not in the `tests/` directory. For example:

- Source: `src/mobius/rewrite_rules/_packed_attention.py`
- Test: `src/mobius/rewrite_rules/_packed_attention_test.py`

This keeps rewrite rule tests co-located with their implementation since
they test internal graph transformation patterns rather than public API
behavior.

### Pattern: adding a new model type (L1)

Add an entry to the appropriate list in `tests/_test_configs.py` with the
model type, config overrides, and `is_representative` flag:

```python
# In tests/_test_configs.py:
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    ("llama", {}, True),
    ("my_model", {"attn_qkv_bias": True, "hidden_act": "gelu"}, True),
    # ...
]
```

The test framework automatically creates a tiny config, builds the graph,
and checks:
- Graph has inputs (`input_ids`, `attention_mask`, `position_ids`)
- Graph has outputs (`logits`, `present.{i}.key`, `present.{i}.value`)
- Graph has initializers (embedding, attention, MLP/expert parameters)

### Pattern: model-specific structure tests

For models with unique structure (e.g. LoRA), add a dedicated test class:

```python
class TestBuildGraphLoRA:
    def test_lora_initializers_present(self):
        config = _base_config(
            vision_lora={"r": 4, "lora_alpha": 8},
            speech_lora={"r": 8, "lora_alpha": 16},
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = CausalLMTask()
        model = task.build_graph(module, config, opset_version=23)

        init_names = list(model.graph.initializers)
        lora_names = [n for n in init_names if "lora" in n]
        assert len(lora_names) > 0
```

---

## L1 structure tests (weight alignment)

Located in `tests/weight_alignment_test.py`. Verifies that
`preprocess_weights()` maps HuggingFace state dict keys to ONNX initializer
names correctly — no dropped or mangled weight names.

For each model type, the test:
1. Builds the ONNX graph with a tiny config
2. Collects all ONNX initializer names
3. Creates a synthetic state dict matching those names
4. Runs `preprocess_weights()` on the dict
5. Asserts all initializers are still covered

This catches bugs like Falcon's `h.` prefix replacement corrupting names,
MoE fused weight names being dropped, or OPT bias names being mangled.

---

## L2: Config compatibility

L2 is detected from the `test_model_id` field in the YAML test case
(see YAML format below).  If a model has a `test_model_id`, the dashboard
counts it as L2.  A model without a YAML case (or without `test_model_id`)
is not counted at L2.

To add L2 coverage: create a YAML test case with `test_model_id` set to a
real HuggingFace model ID (see YAML format section below).

---

## L3: Integration tests (synthetic parity)

Located in `tests/integration_test.py`. Require real HuggingFace checkpoints.
Each model is parametrized with `(model_id, trust_remote_code)`.

### Adding a new model to integration tests

Add a `pytest.param` to `_TEXT_MODELS`:

```python
_TEXT_MODELS = [
    pytest.param("Qwen/Qwen2.5-0.5B", False, id="qwen2.5-0.5b"),
    pytest.param("my-org/my-small-model", False, id="my-model"),
    # ...
]
```

Guidelines for choosing models:
- Prefer models ≤ 1B parameters for CI speed
- Models must be publicly accessible (no gated/private repos)
- One representative model per distinct model class

### Pattern: prefill + decode numerical comparison

```python
@pytest.mark.integration
@pytest.mark.parametrize("model_id,trust_remote_code", _TEXT_MODELS)
class TestForwardNumerical:
    def test_prefill_logits_match(self, model_id, trust_remote_code):
        onnx_model = build(model_id, load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        # Tokenize, run both models, compare
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_outputs = session.run(feeds)
        assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)

    def test_decode_step_logits_match(self, model_id, trust_remote_code):
        # Prefill first, then feed next token + KV cache
        decode_feeds = _make_decode_feeds(config, ...)
        onnx_out_2 = session.run(decode_feeds)
        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)
```

### Pattern: greedy generation

```python
@pytest.mark.integration
class TestGreedyGeneration:
    def test_generate_tokens_match(self, model_id, trust_remote_code):
        session = OnnxModelSession(onnx_model)
        generator = OnnxGenerator(session, config)
        onnx_ids = generator.generate(input_ids, max_new_tokens=10, eos_token_id=...)

        torch_ids = torch_generate_greedy(torch_model, input_ids, max_new_tokens=10, eos_token_id=...)
        assert_generation_match(onnx_ids[0].tolist(), torch_ids[0].tolist())
```

---

## L4 + L5: Golden tests

L4 and L5 tests compare ONNX model outputs against HuggingFace reference
outputs using pre-computed "golden" files stored in `testdata/golden/`.

| Level | Golden file | Contents |
|-------|-------------|----------|
| L4 | `testdata/golden/<cat>/<model>.json` | Prefill top-1/top-2 token IDs + logit summary |
| L5 | `testdata/golden/<cat>/<model>_generation.json` | Prompt + generated token IDs + generated text |

### YAML test case format

**Location:** `testdata/cases/<category>/<model>.yaml`

Categories match task types: `causal-lm`, `encoder`, `seq2seq`, `audio`,
`vision`, `vision-language`, `diffusion`.

**Required fields:**

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # HuggingFace model ID
revision: "main"                            # Git revision / commit SHA
task_type: "text-generation"               # Task type string
dtype: "float32"                           # "float32", "float16", or "bfloat16"
level: "L4+L5"                             # "L4", "L5", or "L4+L5"

inputs:
  prompts:
    - "Here is my poem:"                   # Text prompt(s); use this default
```

For image models, use `images:` instead of (or alongside) `prompts:`:

```yaml
inputs:
  images:
    - "pipeline-cat-chonk.jpeg"            # Path relative to testdata/
```

For audio models:

```yaml
inputs:
  audio:
    - "652-129742-0006.flac"
```

**Optional fields:**

```yaml
# Identifier for the test model used in L2 config compatibility check.
# If set, the dashboard counts this model as L2 (full HF config valid).
test_model_id: "Qwen/Qwen2.5-1.5B-Instruct"

# Skip this test case entirely (model too large, gated repo, etc.).
# Dashboard shows the model as 'skipped' rather than counting it toward coverage.
skip_reason: "Model too large (47B MoE) for CPU golden generation."

# Pass trust_remote_code=True when loading HuggingFace model (default: false).
trust_remote_code: true

# Minimum fraction of generated tokens that must match the golden reference.
# Use for VL/audio pipelines where floating-point variance causes later tokens
# to diverge. A value of 0.25 means at least 25% of tokens must match exactly.
# Green (≥0.9) / Yellow (0.5–0.9) / Red (<0.5) on dashboard.
min_token_match_ratio: 0.25

# Human-readable notes about this model.
notes: "GPT-2 124M. Absolute positional embeddings, no RoPE."

generation:
  max_new_tokens: 20                       # Override token generation limit
  do_sample: false
```

**`skip_reason` vs `_SKIP_REASONS` dict:** Always use the YAML `skip_reason`
field for new cases. The legacy `_SKIP_REASONS` dict in `e2e_golden_test.py`
has been removed — YAML is the canonical location.

### Golden file format

**L4 golden file** (`testdata/golden/<cat>/<model>.json`):
Generated automatically by `generate_golden.py`. Contains `top1_id`,
`top2_id`, `top10_ids`, `top10_logits`, and `logits_summary` from the last
token position of the prefill pass.

**L5 generation file** (`testdata/golden/<cat>/<model>_generation.json`):
Contains `model_id`, `prompt`, `generated_tokens` (list of token IDs), and
`generated_text`. This is the authoritative source for L5 tests — the main
golden JSON does **not** contain generation data.

### Generating golden data

```bash
# Generate for all test cases at a given level
python scripts/generate_golden.py --level L4

# Generate for a specific task type
python scripts/generate_golden.py --level L4 --task-type causal-lm

# Generate for a specific model (glob filter on model name)
python scripts/generate_golden.py --level L4 --filter 'llama*'
```

Golden files must be committed alongside new test case YAML files.

### Running L4/L5 tests

```bash
# L4 tests (single forward pass parity)
python -m pytest tests/e2e_golden_test.py -m golden --level L4 -v

# L5 tests (multi-token generation parity)
python -m pytest tests/e2e_golden_test.py -m golden --level L5 -v
```

### Adding coverage for a new model (step by step)

**L1 — Graph builds:**
1. Add `("my_model", {config_overrides}, True)` to the appropriate list in
   `tests/_test_configs.py` (or add a dedicated method if the model is a VLM/audio).
2. Run `python -m pytest tests/build_graph_test.py -k "my_model"`.

**L2 — Config compatible:**
1. Create `testdata/cases/<category>/my-model.yaml`.
2. Set `test_model_id: "org/my-model-id"`.
3. Run schema validation: `python -m pytest tests/yaml_schema_test.py`.

**L3 — Synthetic parity:**
1. Add `pytest.param("org/my-model", False, id="my-model")` to the
   appropriate parametrized list in `tests/integration_test.py`.
2. Run `python -m pytest tests/integration_test.py -m integration -k "my-model"`.

**L4 — Golden match:**
1. Create/update `testdata/cases/<category>/my-model.yaml` with `level: "L4"`.
2. Set `inputs.prompts: ["Here is my poem:"]` (standard default prompt).
3. Run `python scripts/generate_golden.py --level L4 --filter 'my-model*'`.
4. Commit the generated `testdata/golden/<cat>/my-model.json`.
5. Run `python -m pytest tests/e2e_golden_test.py -m golden --level L4 -k "my-model"`.

**L5 — Generation verified:**
1. Update YAML to `level: "L5"` or `"L4+L5"`.
2. Add a `generation:` block with `max_new_tokens` and `do_sample: false`.
3. Optionally set `min_token_match_ratio` if you expect partial divergence
   (VL pipelines, long generation sequences).
4. Run `python scripts/generate_golden.py --level L5 --filter 'my-model*'`.
5. Commit `testdata/golden/<cat>/my-model_generation.json`.
6. Run `python -m pytest tests/e2e_golden_test.py -m golden --level L5 -k "my-model"`.

### Dashboard coverage

The dashboard shows L4/L5 coverage per model. A model shows as 'skipped'
(not counted toward coverage) when its YAML test case has a `skip_reason`
field. Models without a YAML case show no L4/L5 coverage.

---

## Tolerances

| Test type | Recommended rtol/atol |
|-----------|----------------------|
| Standard text models | `1e-3` / `1e-3` |
| Encoder-only (BERT) | `1e-3` / `1e-3` |
| Encoder-decoder (Whisper, BART, T5) | `1e-3` / `1e-3` |
| Multimodal models | `1e-2` / `1e-2` |
| Diffusion models (UNet, DiT, VAE) | `1e-3` / `1e-3` |
| Audio encoder models | `1e-3` / `1e-3` |
| Generation (token IDs) | Exact match |

Multimodal models use looser tolerances because the vision pipeline
introduces additional floating-point variance.

`assert_logits_close` uses `strict=True` in `np.testing.assert_allclose`,
which also checks shape and dtype match. If tolerances fail, verify:

1. **Norm epsilon** — LayerNorm/RMSNorm eps must match HF config exactly
   (e.g., Whisper uses `1e-5`, not the default `1e-6`)
2. **Norm type** — Check if the model uses RMSNorm or LayerNorm.  OLMo-1B
   uses weight-free LayerNorm (not RMSNorm).  Using the wrong type causes
   max abs diff > 1.0.
3. **Q scaling order** — some models (Whisper) pre-scale Q before attention
   and pass `scale=1.0` to the op, which is numerically different from
   passing `scale=head_dim**-0.5`
4. **Attention scale** — some models (Granite) replace `1/sqrt(head_dim)` with
   a custom `attention_multiplier` from the config
5. **Scaling multipliers** — check HF config for `embedding_multiplier`,
   `logits_scaling`, `residual_multiplier` that aren't in standard Llama
6. **Residual pattern** — verify `residual + output * scale` vs
   `residual * scale + output` by reading HF source
7. **Weight loading** — compare ONNX initializers against HF state_dict to
   rule out name mapping bugs
8. **Float64 contamination** — numpy arrays created from config values default
   to float64; always use `dtype=np.float32`

### Debugging large logit differences

When max abs diff is large (> 0.5), run this diagnostic:

```python
import numpy as np
diff = np.abs(onnx_logits[0, -1] - hf_logits)
print(f"Max abs diff: {diff.max():.4f}")
print(f"Mean abs diff: {diff.mean():.4f}")
# If max > 0.5, it's likely a norm or scaling bug, not just floating-point
# If max > 10, weights are probably loaded to wrong parameters
```

Check the HF norm class directly:
```python
import inspect
from transformers.models.olmo.modeling_olmo import OlmoLayerNorm
print(inspect.getsource(OlmoLayerNorm))
```

Check for unextracted config fields:
```python
config = AutoConfig.from_pretrained("model-id")
for k, v in config.to_dict().items():
    if any(s in k for s in ("multiplier", "scaling", "factor", "epsilon")):
        print(f"{k}: {v}")
```

## Testing utilities reference

| Utility | Import path | Purpose |
|---------|-------------|---------|
| `OnnxModelSession(model)` | `_testing.ort_inference` | Save + load + run ONNX model |
| `OnnxGenerator(session, config)` | `_testing.generation` | Multi-step greedy decoding |
| `load_torch_model(id)` | `_testing.torch_reference` | Load HF model + tokenizer |
| `torch_forward(model, ...)` | `_testing.torch_reference` | Single forward pass |
| `torch_generate_greedy(...)` | `_testing.generation` | Multi-token generation |
| `assert_logits_close(a, b)` | `_testing.comparison` | Logit comparison with diagnostics |
| `assert_generation_match(a, b)` | `_testing.comparison` | Token-ID exact match |

## Debugging multi-model pipelines (TTS, VLM)

When a multi-model pipeline produces wrong output but individual
model prefill logits look correct, isolate each model boundary:

1. **Compare each model's output against HF at the boundary** — e.g.
   `last_hidden_state` from the talker, `codec_sum` from embeddings,
   `inputs_embeds` constructed for the code predictor.

2. **Check pre-norm vs post-norm** — `outputs.last_hidden_state` in HF
   is typically post-norm. If your ONNX model returns pre-norm hidden
   states, downstream models receive wrong values.

3. **Verify external construction matches HF** — for models where the
   generation loop constructs inputs externally (e.g. concatenating
   hidden states with embeddings), write a comparison script that
   checks the constructed input matches HF token-by-token:
   ```python
   # Compare inputs_embeds at each generation step
   for step in range(num_steps):
       onnx_input = construct_inputs_embeds(step, ...)
       hf_input = hf_model.get_inputs_embeds(step, ...)
       diff = np.abs(onnx_input - hf_input).max()
       print(f"Step {step}: max diff = {diff:.6f}")
   ```

4. **Embedding weight vs lookup mismatch** — if embedding weights are
   identical but lookups differ, the issue is usually which code index
   or embedding table is being used (off-by-one errors).

## QA pitfalls: lessons from Qwen3.5 / hybrid-attention models

These lessons come from debugging a hybrid DeltaNet + full-attention model
(Qwen3.5). They apply to any model that uses ONNX custom functions, Scan
ops, or non-standard dtypes.

### Build graph tests are necessary but not sufficient

Build graph tests (L1) verify graph construction — I/O shapes,
initializer existence, no obvious op errors. They do **not** execute the
graph with real data. A Scan body MatMul shape mismatch that would crash at
runtime can still pass all L1 tests. **Always write an integration test
alongside any new custom function or Scan op.**

### Integration tests must exercise all code paths

- **Text-only first** — verify generation and logit parity before adding
  other modalities
- **Vision with real pixel values** — passing empty features (zeros) doesn't
  exercise the vision encoder; use `processor(images=image)` outputs
- **All dtypes**: f32, f16, bf16. Each can expose different bugs:
  - f16/bf16 have different overflow points and precision characteristics
  - Kernel dispatch is dtype-specific (some kernels only exist for f16)
- **GPU when available** — different kernels activate on CUDA; a model
  correct on CPU may diverge on GPU due to different reduction order

### Test feed creation: symbolic dimensions need real values

ONNX models export symbolic batch/sequence dimensions. When feeding the
model for ORT inference:

- **Recurrent state batch dim must match input batch dim** — unlike KV
  cache (which initialises to zeros and grows), recurrent state tensors
  have a fixed `(B, ...)` shape. Feeding batch=0 produces a zero-sized
  carry state that collapses the Scan output.
- **Scan carry state is not KV cache** — do not copy the KV cache
  zero-initialisation pattern for recurrent state; the batch dimension
  must be the actual inference batch size.

```python
# WRONG — batch=0 zeros out Scan carry
past_state = np.zeros((0, num_heads, d_k, d_v), dtype=np.float32)

# CORRECT — must match actual batch size
batch_size = input_ids.shape[0]
past_state = np.zeros((batch_size, num_heads, d_k, d_v), dtype=np.float32)
```

### ONNX function registration

When renaming a custom function's `op_type` (e.g. `CausalConvNdWithState`
→ `CausalConvWithState`), the function must be re-registered under the new
name in ORT's function decomposition list. ORT needs the function embedded
in `model.functions` to decompose the custom op before execution.

**Checklist when renaming a custom function:**
1. Rename the Python factory function and the `ir.Function.name`
2. Update all call sites that reference the old op_type string
3. Update any integration tests that check the op_type name
4. Verify the function appears in `onnx_model.functions` after build

### Dtype-specific bugs to watch for

**fp16 Exp overflow:** `exp(x)` overflows to `inf` for `x > ~11.09` in
fp16. The Softplus activation (`log(1 + exp(x))`) and decay computation
`exp(-softplus(x))` are common overflow sites. Always upcast to float32
for Exp/Softplus in fp16 models. bf16 has the same 8-bit exponent range as fp32 and does NOT need the upcast:
```python
x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
result = op.Exp(x_f32)
result = op.Cast(result, to=x.dtype)  # cast back
```

**bf16 vs fp16:** bf16 has the same exponent range as fp32 (no overflow
at 11.09) but much less precision (7-bit mantissa vs 10-bit). If a
computation works in bf16 but not fp16, check for Exp overflow first.

### Examples as QA tools

The `--compare-hf` flag in example scripts is the gold-standard correctness
check for a model. Run it as part of every significant change:

```bash
# Primary correctness check
python examples/qwen35_text_generation.py --compare-hf

# Test all supported dtypes
python examples/qwen35_text_generation.py --compare-hf --dtype f16
python examples/qwen35_text_generation.py --compare-hf --dtype bf16

# Test on GPU (if available)
python examples/qwen35_text_generation.py --compare-hf --device cuda
```

Target: **100% token match** in fp32 greedy generation. fp16/bf16 may
diverge after the first few tokens due to floating-point accumulation, which
is acceptable if logit parity holds at `atol=rtol=1e-2`.

### Parity testing methodology

Compare full logit tensors, not just generated tokens. Generated tokens
hide logit divergence (two very-different logit vectors can agree on the
top-1 token):

```python
# Always compare full logits at every position
assert_logits_close(onnx_logits, hf_logits, atol=1e-3, rtol=1e-3)  # fp32
assert_logits_close(onnx_logits, hf_logits, atol=1e-2, rtol=1e-2)  # fp16/bf16

# Also check last-position argmax matches (quick sanity check)
assert onnx_logits[0, -1].argmax() == hf_logits[0, -1].argmax()
```

If argmax matches but full logit tolerance fails, the model is numerically
correct but some intermediate accumulation differs — this is usually
acceptable for fp16/bf16 and worth a brief comment in the test.

### Automated code review catches real bugs

Enable Copilot/automated review on every PR that modifies model or component
code. During Qwen3.5 development, automated review found:

- The fp16 Exp overflow risk in decay computation (a real runtime bug on fp16
  hardware that would not be caught by fp32 tests)
- Missing input validation (`kernel_size < 1`, `channels <= 0`) that would
  cause ZeroDivisionError or silent wrong output

These were not caught by the 514 build graph tests. Code review + integration
tests together cover the gaps that unit tests cannot.
