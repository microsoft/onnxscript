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
behaviour.  The project has a 4-tier test pyramid, plus shared configuration
infrastructure.

## Test pyramid

| Tier | File | Runs in CI | Purpose |
|------|------|-----------|---------|
| **0 — Smoke** | `tests/build_graph_test.py` | ✅ Always | Graph builds, I/O shapes, initializers exist |
| **1 — Structure** | `tests/weight_alignment_test.py` | ✅ Always | `preprocess_weights()` covers all initializers |
| **2 — Numerical** | *(planned — manual pattern)* | ❌ Manual | ORT inference with uniform/random weights |
| **3 — Integration** | `tests/integration_test.py` | Opt-in (`-m integration`) | Real HF weights, numerical parity, greedy generation |

### Running tests

```bash
# All non-integration tests (fast, no downloads)
pytest tests/build_graph_test.py tests/cli_test.py src/ -q \
  -k "not phi4mm and not apply_weights_unknown" --tb=short

# Representative models only (~5 seconds)
pytest tests/build_graph_test.py --fast

# Single model type
pytest tests/build_graph_test.py -k "phi4mm"

# Integration tests (slow, downloads models)
pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"
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
pytest tests/build_graph_test.py -k "my_model"
```

---

## Tier 0: Smoke tests (graph construction)

Located in `tests/build_graph_test.py`. Uses tiny synthetic
`ArchitectureConfig` objects (64 hidden, 2 layers, 256 vocab) to test
every model type without downloading weights or requiring network access.
Configs are defined in `tests/_test_configs.py` (see above).

### Rewrite rule unit tests

Rewrite rule unit tests should be placed **next to** the source file they
test, not in the `tests/` directory. For example:

- Source: `src/mobius/rewrite_rules/_packed_attention.py`
- Test: `src/mobius/rewrite_rules/_packed_attention_test.py`

This keeps rewrite rule tests co-located with their implementation since
they test internal graph transformation patterns rather than public API
behavior.

### Pattern: adding a new model type

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

## Tier 1: Structure tests (weight alignment)

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

## Tier 2: Numerical tests (planned)

Not yet implemented as a formal test tier.  The pattern below describes
a manual workflow for developing a new model or debugging a mismatch:
build the ONNX model with random/uniform weights, run ORT inference,
and check outputs are finite and reasonable.  No on-demand scripts
exist yet — follow this pattern ad-hoc when needed.

---

## Tier 3: Integration tests

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
