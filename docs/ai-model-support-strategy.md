# Strategy: AI-Assisted Model Support

This document describes the strategy for using AI agents (e.g. GitHub Copilot,
LLM-based coding assistants) to add support for new model architectures in
`mobius`. It covers three categories:

1. **New HuggingFace Transformers models** — models in the `transformers` library
2. **New Diffusers models** — pipelines and components from the `diffusers` library
3. **Out-of-library models** — architectures not in either library

---

## Motivation

The generative AI landscape produces new model architectures faster than any
team can manually keep up with. By codifying our patterns into skills, tests,
and structured metadata, we enable AI agents to:

- **Classify** incoming model requests automatically
- **Generate** model classes, configs, and weight mappings from HF source code
- **Validate** correctness through integration tests against PyTorch reference
- **Scale** support to hundreds of architectures with minimal human review

---

## Architecture recap

```
HuggingFace source code
       │
       ▼
┌─────────────────┐     ┌──────────────────────┐
│ Classification   │────▶│ Pattern matching      │
│ (model_type,     │     │ (which base class?    │
│  task, category) │     │  which components?)   │
└─────────────────┘     └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ Code generation       │
                        │ (model file, config,  │
                        │  registry, weights)   │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ Validation            │
                        │ (unit test, integration│
                        │  test, generation)     │
                        └──────────────────────┘
```

Each step is described in detail below, with specific instructions for AI
agents.

---

## Phase 1: Classification

Given a model identifier (e.g. `meta-llama/Llama-4-Scout-17B-16E`), the AI
agent must first classify the model to determine the implementation strategy.

### Step 1.1 — Determine the source library

| Signal | Library | Detection method |
|--------|---------|-----------------|
| `config.json` has `model_type` | Transformers | `AutoConfig.from_pretrained()` succeeds |
| `model_index.json` exists | Diffusers | `hf_hub_download("model_index.json")` succeeds |
| Neither | Out-of-library | Fall through to manual analysis |

### Step 1.2 — Determine the task type

For **Transformers** models, read the HuggingFace model card and `config.json`:

| Indicator | Task | Base class |
|-----------|------|------------|
| `model_type` is decoder-only LLM | `text-generation` | `CausalLMModel` |
| Has `vision_config` or image processor | `vision-language` | `LLaVAModel` or custom multimodal |
| Has `encoder`/`decoder` structure | `seq2seq` | `BartForConditionalGeneration` / `T5ForConditionalGeneration` |
| Encoder-only (BERT-like) | `feature-extraction` | `BertModel` |
| Speech/audio model | `speech-to-text` or `audio-feature-extraction` | `WhisperForConditionalGeneration` / `Wav2Vec2Model` |
| Vision classifier | `image-classification` | `ViTModel` |

For **Diffusers** models, read `model_index.json`:

| Component class | Task | Base class |
|----------------|------|------------|
| `*Transformer2DModel` | `denoising` | Closest existing: `FluxTransformer2DModel`, `SD3Transformer2DModel`, `DiTTransformer2DModel` |
| `AutoencoderKL*` | `vae` | `AutoencoderKLModel` or `AutoencoderKLQwenImageModel` |
| `UNet2DConditionModel` | `denoising` | `UNet2DConditionModel` |
| `ControlNet*` | `controlnet` | `ControlNetModel` |

### Step 1.3 — Classify the implementation effort

| Level | Description | Action |
|-------|-------------|--------|
| **Trivial** | Standard Llama-like decoder (RoPE, GQA, RMSNorm, SiLU MLP) | Register `CausalLMModel` directly; no new code |
| **Config-only** | Same architecture, different config field names | Update `from_transformers()` extraction logic |
| **Variant** | Minor architectural difference (custom norm, scaling, bias) | Subclass base model/component; ~50–200 lines |
| **New architecture** | Fundamentally different (new attention pattern, new task type) | Full model implementation; ~200–600 lines |
| **New paradigm** | New task type or I/O contract | New task class + model; may need new components |

**AI agent instruction:** Always start by attempting Level 0 (trivial). Build
the model with `CausalLMModel`, run integration test. If tokens match, done.
Only escalate if the test fails.

---

## Phase 2: Analysis (AI-driven)

Once classified, the AI agent analyses the HuggingFace source code to determine
what implementation is needed.

### Step 2.1 — Read the HuggingFace source

The agent should read the PyTorch model source code. For Transformers models:

```python
# Locate the HF source file for the model_type
import transformers
import inspect
auto_class = transformers.AutoModelForCausalLM  # or appropriate Auto class
model_class = auto_class._model_mapping[config.__class__]
source_file = inspect.getfile(model_class)
```

For Diffusers models:

```python
import diffusers
component_class = getattr(diffusers, class_name)
source_file = inspect.getfile(component_class)
```

### Step 2.2 — Compare against existing components

The agent should systematically compare each part of the HF model against the
existing component library:

| HF component | Compare against | Key questions |
|-------------|----------------|---------------|
| `__init__` constructor | `CausalLMModel.__init__` | Same layer structure? Same norm type? |
| Attention class | `components.Attention` | Different head arrangement? Custom scaling? QK norm? |
| MLP class | `components.MLP` | Different gating? Different activation? |
| Normalization | `components.RMSNorm` / `LayerNorm` | Which norm? Custom offset? Weight-free? Different eps? |
| Positional encoding | `components.initialize_rope` | Standard RoPE? LongRope? MRoPE? ALiBi? Learned? |
| Embedding | `components.Embedding` | Scaling? Absolute position? |
| `forward()` method | `CausalLMModel.forward` | Residual pattern? Extra multipliers? |

**AI agent instruction:** For each component, classify as:
- **Identical** → reuse existing component
- **Configurable** → reuse with different config values
- **Variant** → subclass and override specific behavior
- **Novel** → implement new component

### Step 2.3 — Map weight names

Compare HuggingFace `state_dict` keys against the ONNX model's
`named_parameters()`:

```python
# 1. Build ONNX model without weights
module = MyModel(config)
onnx_names = set(n for n, _ in module.named_parameters())

# 2. Load HF state dict
from safetensors import safe_open
hf_names = set(state_dict.keys())

# 3. Find mismatches
only_hf = hf_names - onnx_names    # Need renaming in preprocess_weights
only_onnx = onnx_names - hf_names  # Need weight tying or generation
```

Common patterns the agent should recognise:

| Pattern | Example | preprocess_weights action |
|---------|---------|--------------------------|
| Prefix strip | `language_model.model.X` → `model.X` | `key.replace("language_model.", "")` |
| QKV split | `qkv_proj.weight` (3H×H) | Split into `q_proj`, `k_proj`, `v_proj` |
| Gate/up split | `gate_up_proj.weight` (2I×H) | Split into `gate_proj`, `up_proj` |
| Expert rename | `experts.0.w1` | → `experts.0.gate_proj` |
| Weight tying | Missing `lm_head.weight` | Copy from `embed_tokens.weight` |
| Norm rename | `weight` vs `gamma` | Match HF naming via `nn.Parameter` name |

---

## Phase 3: Generation

### Strategy A — Transformers models

#### A.1. Trivial (register-only)

```python
# In _create_default_registry() in _registry.py:
reg.register("new_model_type", CausalLMModel)
```

No model file needed. Add a build_graph test entry and integration test.

#### A.2. Config-only

Update `ArchitectureConfig.from_transformers()` in `_configs.py` to extract
model-specific fields:

```python
# In from_transformers():
config.my_special_field = getattr(hf_config, "my_special_field", default_value)
```

Add the field to the `ArchitectureConfig` dataclass.

#### A.3. Variant (subclass)

Create a new model file. The agent should follow the **adding-a-new-model**
skill (`.github/skills/adding-a-new-model/SKILL.md`) which provides:

1. A complete model file template
2. Common variation patterns with solutions
3. A troubleshooting guide for common pitfalls
4. Weight mapping strategies

**Key principle:** Subclass, don't flag. Create a new `MyRMSNorm(RMSNorm)` or
`MyDecoderLayer(DecoderLayer)` rather than adding boolean flags to existing
components.

#### A.4. New architecture

For fundamentally different architectures, the agent should:

1. Identify which existing components can be reused
2. Create new components for novel parts (follow **reusable-components** skill)
3. Compose into a model class
4. Wire into an appropriate task (or create a new task if needed)

### Strategy B — Diffusers models

Diffusers models have a different structure than Transformers:

| Aspect | Transformers | Diffusers |
|--------|-------------|-----------|
| Config source | `AutoConfig` → `config.json` | `model_index.json` → per-component `config.json` |
| Detection | `model_type` string | `_class_name` in `model_index.json` |
| Weight files | `model.safetensors` | `diffusion_pytorch_model.safetensors` |
| Registry | `ModelRegistry` by `model_type` | `_DIFFUSERS_CLASS_MAP` by `_class_name` |
| Config class | `ArchitectureConfig` | Domain-specific (`VAEConfig`, `UNet2DConfig`, etc.) |

#### B.1. Adding a new diffusers component

1. **Create a config dataclass** in `_diffusers_configs.py`:

```python
@dataclasses.dataclass
class MyDiffuserConfig:
    field1: int
    field2: float

    @classmethod
    def from_diffusers(cls, config: dict) -> MyDiffuserConfig:
        return cls(field1=config["field1"], field2=config.get("field2", 1.0))
```

2. **Create the model module** in `models/my_diffuser.py`

3. **Register in `_DIFFUSERS_CLASS_MAP`** (in `_diffusers_builder.py`):

```python
_DIFFUSERS_CLASS_MAP["MyDiffuserClass"] = (MyDiffuserModel, MyDiffuserConfig, "denoising")
```

4. **Create or reuse a task** from `tasks/`

#### B.2. Adding a new diffusers pipeline

A pipeline is a collection of components. The `build_diffusers_pipeline()`
function iterates `model_index.json` and builds each recognised component.
To support a new pipeline, ensure all its components are registered in
`_DIFFUSERS_CLASS_MAP`.

### Strategy C — Out-of-library models

For models not in Transformers or Diffusers:

1. **Find the source code** (GitHub, paper, reference implementation)
2. **Create a custom config dataclass** (do NOT force-fit `ArchitectureConfig`)
3. **Implement the model using existing components** where possible
4. **Create a custom task** if the I/O contract doesn't match existing tasks
5. **Write a custom weight loader** if weights aren't in standard safetensors format

The agent should look for structural similarities with existing models:

| If the model looks like... | Start from... |
|---------------------------|---------------|
| Decoder-only transformer | `CausalLMModel` / `TextModel` |
| Encoder-decoder | `BartForConditionalGeneration` / `T5ForConditionalGeneration` |
| Vision transformer | `ViTModel` / `VisionModel` component |
| Diffusion model | `UNet2DConditionModel` / `DiTTransformer2DModel` |
| VAE | `AutoencoderKLModel` |
| CNN-based | Create new components using `_Conv2d`, `_Conv3d` patterns from `qwen_image_vae.py` |

---

## Phase 4: Validation

### 4.1. Unit tests (mandatory)

Add a tiny config entry to `tests/build_graph_test.py`:

```python
("new_model_type", {"hidden_act": "silu"}),
```

This verifies the model graph builds without errors using a minimal synthetic
config (64 hidden dimensions, 2 layers, no weights).

### 4.2. Integration tests (mandatory for new architectures)

Add the smallest available checkpoint to `tests/integration_test.py`:

```python
pytest.param("org/model-name", False, id="model-name"),
```

This downloads real weights, builds the ONNX model, and compares logits against
the HuggingFace PyTorch reference. Acceptance criteria:

| Check | Tolerance |
|-------|-----------|
| Prefill logits | `atol=1e-3, rtol=1e-2` (98% of values) |
| Decode logits | Same tolerance |
| Greedy generation | Exact token match for 24 tokens |

### 4.3. Debugging failures

Common failure modes and diagnostic steps:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `tensor(double)` ORT error | Float64 arrays in RoPE/config | Cast to `np.float32` |
| Tokens diverge after position 4-5 | Wrong norm type (RMSNorm vs LayerNorm) | Check HF source for exact norm |
| Large logit mismatch (> 1.0) | Missing scaling multiplier | Check for embed/attention/residual/logit scaling |
| All logits wrong | Weight name mismatch (silently zero) | Compare `named_parameters()` vs `state_dict` keys |
| Shape mismatch error | Wrong head_dim or intermediate_size | Print config values vs HF config |
| `KeyError` in `load_state_dict` | Missing `preprocess_weights` rename | List mismatched keys and add rename rules |

### 4.4. Generation test

After logits match, run greedy generation to verify autoregressive consistency:

```python
prompt = "Once upon a time"
onnx_tokens = onnx_greedy_generate(model, tokenizer, prompt, max_tokens=24)
hf_tokens = hf_greedy_generate(model_id, prompt, max_tokens=24)
assert onnx_tokens == hf_tokens
```

---

## AI Agent Workflow

This section describes the end-to-end workflow an AI agent should follow when
asked to add a new model.

### Inputs

- **Model identifier**: HuggingFace model ID (e.g. `meta-llama/Llama-4-Scout-17B-16E`)
- **Skills**: The agent should read the relevant skill files from `.github/skills/`

### Workflow

```
1. CLASSIFY
   ├─ Download config.json / model_index.json
   ├─ Determine: Transformers? Diffusers? Out-of-library?
   ├─ Determine: task type, category
   └─ Determine: implementation level (trivial → new paradigm)

2. ANALYSE
   ├─ Read HF PyTorch source code
   ├─ Compare each component against existing library
   ├─ Identify: which components to reuse, which to create
   ├─ Map weight names (HF → ONNX)
   └─ List config fields to extract

3. ATTEMPT TRIVIAL
   ├─ Register model_type → CausalLMModel (or closest base)
   ├─ Add build_graph test entry
   ├─ Run unit test
   ├─ Run integration test (if checkpoint available)
   └─ If tokens match → DONE

4. IMPLEMENT (if trivial fails)
   ├─ Read the appropriate skill:
   │   ├─ adding-a-new-model (general)
   │   ├─ moe-models (if MoE)
   │   ├─ multimodal-models (if vision-language)
   │   └─ reusable-components (if new component needed)
   ├─ Create model file
   ├─ Update config extraction if needed
   ├─ Implement preprocess_weights
   ├─ Register in _registry.py and models/__init__.py
   └─ Update build_graph test

5. VALIDATE
   ├─ Run unit tests (all 236+ must pass)
   ├─ Run integration test for new model
   ├─ Run lintrunner
   └─ If failures → diagnose using Phase 4.3 table

6. COMMIT
   ├─ Separate commits for model, config, tests
   └─ Linear commit history (no amends)
```

### Skills reference

| Skill | When to invoke |
|-------|---------------|
| `adding-a-new-model` | Always, for any new model |
| `reusable-components` | When creating a new component (norm, attention variant, etc.) |
| `moe-models` | When the model uses mixture-of-experts |
| `multimodal-models` | When the model processes images + text |
| `writing-tests` | When writing integration tests |
| `writing-rewrite-rules` | When adding post-export graph transformations |

---

## Scaling strategy

### Batch processing

To support a batch of new models, the agent should:

1. **Triage**: Classify all models by implementation level
2. **Trivial first**: Register all trivial models (register-only, config-only)
   in a single commit
3. **Variants next**: Implement variant models grouped by family (e.g. all
   Gemma variants together)
4. **Novel last**: Implement new architectures one at a time with full testing

### Keeping up with new releases

When a new model appears on HuggingFace:

1. The agent downloads `config.json`
2. Checks if `model_type` is already in the registry
3. If not, runs the classification workflow above
4. For most new models (which are Llama-like), this is a trivial register-only
   change

### Coverage metrics

Track model support coverage:

```python
# All model_types seen on HuggingFace Hub
hub_model_types = set(...)  # from HF API

# Supported model_types
supported = set(registry._map.keys())

# Coverage
coverage = len(supported & hub_model_types) / len(hub_model_types)
```

### Priority heuristic

When deciding which unsupported model to add next, prioritise by:

1. **Download count** on HuggingFace Hub (higher = more impactful)
2. **Architectural novelty** (truly new vs. minor variant of existing)
3. **Reusability** (does adding this model also enable a family?)
4. **Test availability** (is there a small checkpoint for validation?)

---

## Appendix: Component reuse matrix

This matrix shows which existing components each model family uses, helping the
agent quickly identify what's reusable for a new architecture.

| Component | Llama-like | Gemma | Falcon | Phi | MoE | BERT | T5 | Whisper | ViT | Diffusion |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Embedding` | ✓ | custom | ✓ | ✓ | ✓ | custom | ✓ | ✓ | custom | — |
| `Linear` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RMSNorm` | ✓ | custom | — | ✓ | ✓ | — | ✓ | — | — | — |
| `LayerNorm` | — | — | ✓ | — | — | ✓ | — | ✓ | ✓ | ✓ |
| `Attention` | ✓ | ✓ | ALiBi | ✓ | ✓ | encoder | enc+dec | custom | encoder | cross |
| `MLP` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `DecoderLayer` | ✓ | custom | custom | ✓ | MoE | — | ✓ | custom | — | — |
| `RoPE` | ✓ | ✓ | — | ✓ | ✓ | — | relative | — | — | — |
| `MoELayer` | — | — | — | — | ✓ | — | — | — | — | — |
| `VisionModel` | — | ✓ | — | ✓ | — | — | — | — | ✓ | — |
| `ConvBlock` | — | — | — | — | — | — | — | — | — | ✓ |
| `TimeEmbedding` | — | — | — | — | — | — | — | — | — | ✓ |
