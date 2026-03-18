# Config System Redesign: mobius

## Problem Statement

`ArchitectureConfig` in `_configs.py` has ~170 fields and a 573-line
`from_transformers()` monolith. Every new model must modify this central
file. The Critical Reviewer found that 66 architectures in the registry
are not in `SUPPORTED_ARCHITECTURES`, making them silently broken.

**Root cause:** Config extraction and config storage are conflated in one
place. The fix is to separate them: keep a lean shared config struct, but
move extraction logic to where the domain knowledge lives — the model
classes.

## Design Principles

1. **Each model owns its config extraction** — no central monolith
2. **Adding a new model never touches `_configs.py`**
3. **Components stay model-agnostic** — they accept the same config interface
4. **Incremental migration** — can move one model at a time
5. **The WhisperConfig and diffusers config patterns are the blueprint**

---

## 1. New Config Class Hierarchy

### Key insight: Don't split the dataclass, split the parser

The exploration revealed that ~40 fields are actively read across the
codebase, with ~20 being truly shared. But the problem isn't the struct —
it's the 573-line function. `ArchitectureConfig` as a flat dataclass is
actually fine for component consumption. The real issue is that
`from_transformers()` contains model-specific parsing logic that belongs
in model files.

### The hierarchy

```
BaseModelConfig                    # ~12 fields: vocab, hidden, heads, kv_heads, head_dim, etc.
├── ArchitectureConfig             # ~55 fields: + rope, norm, attention, MoE, MLA, encoder-decoder
│   ├── (used directly by most models — Llama, Qwen2, Mistral, etc.)
│   ├── FalconConfig               # + alibi, parallel_attn, overrides attn bias logic
│   ├── GraniteConfig              # + scaling multipliers
│   ├── Gemma2Config               # + soft-capping, query_pre_attn_scalar
│   ├── Gemma3nConfig              # + altup/laurel fields
│   ├── MllamaConfig               # + cross_attention_layers
│   ├── YolosConfig                # + detection tokens, num_labels
│   ├── DepthAnythingConfig        # + neck/reassemble/fusion config
│   ├── SegformerConfig            # + hierarchical encoder config
│   └── Sam2Config                 # + hiera backbone config
├── WhisperConfig                  # (already exists — encoder-decoder speech)
├── VisionLanguageConfig           # ArchitectureConfig + VisionConfig (new)
│   ├── (used by LLaVA, Qwen-VL, Mllama, etc.)
│   └── Phi4MMConfig               # + AudioConfig + LoRA + hardcoded SigLIP
├── SpeechLanguageConfig           # ArchitectureConfig + AudioConfig (new)
│   └── (used by Qwen3-ASR)
└── TTSModelConfig                 # ArchitectureConfig + TTSConfig + CodecConfigs (new)
    └── (used by Qwen3-TTS, Qwen3-TTS-Tokenizer)
```

### What changes vs. today

**`ArchitectureConfig` shrinks.** Remove:
- All `vision_*` flat fields (20 fields) → moved to `VisionLanguageConfig`
- All `audio_*` flat fields (10 fields) → moved to `SpeechLanguageConfig`
- `image_crop_size`, `vision_lora`, `speech_lora` → moved to `Phi4MMConfig`
- Falcon-specific: `alibi`, `parallel_attn` → moved to `FalconConfig`
- Granite-specific: `embedding_multiplier`, `attention_multiplier`,
  `logits_scaling`, `residual_multiplier` → moved to `GraniteConfig`
- Gemma2-specific: `attn_logit_softcapping`, `final_logit_softcapping`,
  `query_pre_attn_scalar` → moved to `Gemma2Config`
- Gemma3n-specific: `altup_*`, `laurel_*`, `hidden_size_per_layer_input`,
  `vocab_size_per_layer_input` → moved to `Gemma3nConfig`
- YOLOS: `num_detection_tokens`, `num_labels` → moved to `YolosConfig`
- Depth Anything: `neck_hidden_sizes`, etc. → moved to `DepthAnythingConfig`
- Segformer: `segformer_*` → moved to `SegformerConfig`
- SAM2: `sam2_*` → moved to `Sam2Config`

**`ArchitectureConfig` keeps** (~55 fields):
- Core: hidden_size, vocab_size, intermediate_size, num_hidden_layers,
  num_attention_heads, num_key_value_heads, head_dim
- Attention: attn_qkv_bias, attn_o_bias, attn_qk_norm, attn_qk_norm_full,
  attn_output_gate, sliding_window, layer_types, full_attention_interval
- Norm: rms_norm_eps
- MLP: hidden_act, mlp_bias
- RoPE: rope_type, rope_theta, rope_scaling, partial_rotary_factor,
  rope_local_base_freq, original_max_position_embeddings,
  max_position_embeddings, rope_interleave
- MRoPE: mrope_section, mrope_interleaved
- MoE: num_local_experts, num_experts_per_tok, moe_intermediate_size,
  shared_expert_intermediate_size, norm_topk_prob, decoder_sparse_step,
  n_group, topk_group, routed_scaling_factor, scoring_func, topk_method,
  first_k_dense_replace, n_shared_experts
- MLA: q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
  v_head_dim
- Linear attention: linear_conv_kernel_dim, linear_key_head_dim, etc.
- Encoder-only: type_vocab_size
- Encoder-decoder: num_decoder_layers, relative_attention_num_buckets,
  relative_attention_max_distance
- Standalone vision: image_size, patch_size, num_channels
- Meta: dtype, pad_token_id, tie_word_embeddings
- Nested: vision (VisionConfig), audio (AudioConfig), tts (TTSConfig),
  codec_decoder, codec_encoder

**Note on MoE/MLA fields staying:** These are used by the `_moe.py` and
`_deepseek_mla.py` components, which are model-agnostic. Moving them to
a DeepSeekConfig would mean the components need to know about
DeepSeek-specific configs, violating the layer boundary. Since multiple
model families use MoE (Mixtral, Arctic, Qwen-MoE, DeepSeek, etc.),
these fields belong on the shared config.

---

## 2. How `from_transformers()` Gets Decomposed

### Shallow field extraction helper

**Critical:** `dataclasses.asdict()` recursively converts nested
dataclasses (VisionConfig, AudioConfig) to plain dicts. This silently
breaks any code that does `isinstance(config.vision, VisionConfig)`.
We use a shallow helper instead:

```python
def _shallow_fields(config) -> dict:
    """Extract fields from a dataclass without recursive conversion.

    Unlike dataclasses.asdict(), this preserves nested dataclass
    instances (VisionConfig, AudioConfig, etc.) as-is.
    """
    return {f.name: getattr(config, f.name)
            for f in dataclasses.fields(config)}
```

All subclass `from_transformers()` methods use `_shallow_fields(base)`
instead of `dataclasses.asdict(base)`. This is the single helper that
every contributor must use — getting this wrong silently corrupts nested
configs.

### Current flow

```
build(model_id)
  → _config_from_hf(hf_config, module_class)
    → resolution: module_class.config_class or registry config_class or ArchitectureConfig
    → config_cls.from_transformers(hf_config)   ← THE MONOLITH
```

### New flow

```
build(model_id)
  → _config_from_hf(hf_config, parent_config, module_class)
    → resolution: module_class.config_class or registry config_class or ArchitectureConfig
    → config_cls.from_transformers(hf_config, parent_config)
                  ↓
       ArchitectureConfig.from_transformers()     ← ~120 lines (shared core)
       FalconConfig.from_transformers()           ← calls super() + 8 lines
       Gemma2Config.from_transformers()           ← calls super() + 5 lines
       VisionLanguageConfig.from_transformers()   ← calls super() + vision extraction
       Phi4MMConfig.from_transformers()           ← calls super() + audio + hardcoded SigLIP
```

### The shared core: `ArchitectureConfig.from_transformers()`

This is the 573-line monolith reduced to ~120 lines. It extracts ONLY
the fields that `ArchitectureConfig` defines:

```python
@dataclasses.dataclass
class ArchitectureConfig(BaseModelConfig):
    # ... ~55 fields (see Appendix for full inventory) ...

    @classmethod
    def from_transformers(
        cls, config, parent_config=None, **overrides
    ) -> ArchitectureConfig:
        """Extract shared architecture fields from a HuggingFace config.

        Subclasses should call ``super().from_transformers()`` to get the
        base options dict, then add their own fields::

            @classmethod
            def from_transformers(cls, config, **kw):
                base = ArchitectureConfig.from_transformers(
                    config, **kw
                )
                return cls(
                    **_shallow_fields(base),
                    my_field=getattr(config, "my_field", default),
                )
        """
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_parameters = getattr(config, "rope_parameters", None) or {}

        options = dict(
            # Core architecture (same as today, lines 423-444)
            head_dim=...,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=...,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=...,
            hidden_size=config.hidden_size,
            intermediate_size=...,
            hidden_act=...,

            # Attention config
            attn_qkv_bias=...,
            attn_o_bias=...,
            attn_qk_norm=...,
            sliding_window=...,
            layer_types=...,

            # RoPE config
            rope_type=...,
            rope_theta=...,
            rope_scaling=...,
            partial_rotary_factor=...,

            # MoE config (shared across Mixtral, DeepSeek, Arctic, etc.)
            num_local_experts=...,
            num_experts_per_tok=...,
            # ... etc.

            # MLA config
            q_lora_rank=...,
            kv_lora_rank=...,
            # ... etc.
        )

        # Compute layer_types from full_attention_interval (generic, not model-specific)
        if options.get("layer_types") is None:
            full_attention_interval = options.get("full_attention_interval")
            if full_attention_interval is not None:
                # ... same logic as today ...

        # Model dtype (generic)
        _apply_dtype(options, config)

        options.update(overrides)
        return cls(**options)
```

### Model-specific configs: Example patterns

**FalconConfig** (~30 lines total, lives in `models/falcon.py`):

```python
@dataclasses.dataclass
class FalconConfig(ArchitectureConfig):
    """Configuration for Falcon/Bloom models."""
    alibi: bool = False
    parallel_attn: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None, **kw) -> FalconConfig:
        # Reuse shared extraction — returns ArchitectureConfig
        base = ArchitectureConfig.from_transformers(
            config, parent_config=parent_config, **kw
        )
        model_type = config.model_type

        # Falcon MQA override
        num_kv_heads = base.num_key_value_heads
        if getattr(config, "multi_query", False) and not getattr(
            config, "new_decoder_architecture", False
        ):
            num_kv_heads = 1

        mlp_bias = getattr(config, "bias", False)
        alibi = model_type == "bloom"

        return cls(
            **_shallow_fields(base),
            num_key_value_heads=num_kv_heads,
            mlp_bias=mlp_bias if model_type != "bloom" else True,
            alibi=alibi,
            parallel_attn=getattr(config, "parallel_attn", False),
        )
```

**VisionLanguageConfig** (~80 lines, lives in `_configs.py`):

```python
@dataclasses.dataclass
class VisionLanguageConfig(ArchitectureConfig):
    """Configuration for vision-language multimodal models."""
    vision: VisionConfig | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None, **kw):
        base = ArchitectureConfig.from_transformers(
            config, parent_config=parent_config, **kw
        )

        # Vision sub-config extraction (lines 654-778 of today's monolith)
        vision_source = parent_config or config
        hf_vision = getattr(vision_source, "vision_config", None)
        if hf_vision is None:
            hf_vision = getattr(config, "vision_config", None)

        vision = None
        if hf_vision is not None:
            vc = hf_vision if not isinstance(hf_vision, dict) else type("V", (), hf_vision)()
            vision = VisionConfig(
                hidden_size=getattr(vc, "hidden_size", None),
                intermediate_size=getattr(vc, "intermediate_size", None),
                num_hidden_layers=getattr(vc, "num_hidden_layers", None) or getattr(vc, "depth", None),
                num_attention_heads=getattr(vc, "num_attention_heads", None),
                image_size=getattr(vc, "image_size", None),
                patch_size=getattr(vc, "patch_size", None),
                norm_eps=getattr(vc, "layer_norm_eps", 1e-6),
                out_hidden_size=getattr(vc, "out_hidden_size", None),
                in_channels=getattr(vc, "in_channels", 3),
                spatial_merge_size=getattr(vc, "spatial_merge_size", 2),
                temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
                mm_tokens_per_image=getattr(vision_source, "mm_tokens_per_image", None),
                image_token_id=getattr(vision_source, "image_token_id", None),
            )

        return cls(**_shallow_fields(base), vision=vision)
```

**Phi4MMConfig** (~60 lines, lives in `models/phi.py`):

```python
@dataclasses.dataclass
class Phi4MMConfig(VisionLanguageConfig):
    """Configuration for Phi4-MM multimodal models."""
    audio: AudioConfig | None = None
    vision_lora: dict | None = None
    speech_lora: dict | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None, **kw):
        base = VisionLanguageConfig.from_transformers(
            config, parent_config=parent_config, **kw
        )

        # Override vision with hardcoded SigLIP params
        vision = VisionConfig(
            hidden_size=1152, intermediate_size=4304,
            num_hidden_layers=27, num_attention_heads=16,
            image_size=base.vision.image_crop_size or 448 if base.vision else 448,
            patch_size=14, norm_eps=1e-6,
            image_token_id=getattr(config, "special_image_token_id", 200010),
        )

        # Audio config from audio_processor
        audio = _extract_phi4mm_audio(config)

        return cls(
            **_shallow_fields(base),
            vision=vision,
            audio=audio,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
        )
```

---

## 3. Migration Strategy: Incremental, One Model at a Time

### Phase 0a: Extract shared helpers (1 PR — prerequisite)

Extract reusable parsing logic from `from_transformers()` into helper
functions. This is a pure refactor — no behavior change.

```python
# New helpers in _configs.py (or _config_helpers.py):
def _extract_rope_config(config) -> dict:
    """Extract and normalize rope_scaling/rope_parameters (~30 lines)."""
    ...

def _extract_vision_config(config, parent_config=None) -> dict:
    """Extract VisionConfig fields from HF config (~60 lines)."""
    ...

def _extract_audio_config(config) -> dict:
    """Extract AudioConfig/TTS fields from HF config (~40 lines)."""
    ...
```

After this PR, `from_transformers()` calls these helpers instead of
inline logic. All 267 models pass the same test suite — zero behavior
change. This is the critical prerequisite: model-specific subclasses
will reuse these helpers instead of duplicating shared logic.

### Phase 0b: Infrastructure (1 PR)

1. Add `**overrides` parameter to `ArchitectureConfig.from_transformers()`.
2. Ensure `_config_from_hf()` passes `parent_config` through to
   `from_transformers()`.
3. Delete `SUPPORTED_ARCHITECTURES` (done — commit 683a752).

This is backward-compatible. All existing code keeps working.

### Phase 1: Extract model-specific branches (1 PR per model family)

For each model family with `if model_type == "..."` branches in
`from_transformers()`:

1. Create a config subclass in the model file (e.g. `FalconConfig` in
   `models/falcon.py`)
2. Move the model-specific parsing logic to its `from_transformers()`
3. Set `config_class = FalconConfig` on the model class
4. Remove the `if model_type` branch from `ArchitectureConfig.from_transformers()`
5. The registry already supports `config_class` — no registry changes needed

**Only 3 model families need custom config subclasses** (all others work
with `ArchitectureConfig` as-is):

1. **Falcon/Bloom/MPT family** → `FalconConfig` (~15 lines)
   - One config class handles all 4 model_types (falcon, falcon_h1, bloom, mpt)
   - All 4 registered to `FalconCausalLMModel` in _registry.py:263-268
   - Set `config_class = FalconConfig` on `FalconCausalLMModel`
   - Handles: MQA quirk for falcon/falcon_h1, alibi for bloom, mlp_bias
2. **Vision/Audio sub-config** → `VisionLanguageConfig` (~60 lines)
   - Calls `_extract_vision_config()` helper from Phase 0a
   - Handles ~200 lines of vision parsing currently in monolith
3. **Phi4MM** → `Phi4MMConfig` (~40 lines, extends VisionLanguageConfig)
   - Hardcoded SigLIP vision params, audio config, LoRA fields
4. **Qwen3 TTS** → `Qwen3TTSConfig` (~30 lines)
   - Decoder/encoder config extraction for tokenizer_12hz

**Simple models (Llama-compatible) do NOT need custom configs.** The ~40
architectures using CausalLMModel (llama, mistral, qwen2, phi3,
internlm2, etc.) continue using `ArchitectureConfig.from_transformers()`
unchanged. No one is forced to write a custom config class unless their
HuggingFace config has quirks requiring model-specific parsing.

After Phase 1: `from_transformers()` is ~120 lines of shared extraction.

### Phase 2: Migrate flat vision_* fields to config.vision.* (3-5 PRs)

**Scope correction:** The flat `vision_hidden_size`, `vision_num_attention_heads`,
etc. fields have **132 references across 16 files** — this is NOT a 1-PR
change. Files affected include:

- `components/_vision.py` (15 refs)
- `components/_multimodal.py` (9 refs)
- `models/qwen_vl.py` (24 refs)
- `models/phi.py` (11 refs)
- `models/gemma3.py` (4 refs)
- `models/llava.py` (1 ref)
- `tasks/_vision_language.py` (1 ref)
- `tasks/_vision_language_3model.py` (2 refs)
- `tasks/_multimodal.py` (1 ref)
- `_configs.py` (14 refs — field definitions + from_transformers)
- 6 test files (~44 refs combined)

**Strategy:** Add `@property` aliases on `ArchitectureConfig` that
delegate to `self.vision.*` (see Section 6 below), then migrate
consumers one file at a time. Remove the aliases only after all
consumers are updated.

Estimated: 3-5 PRs, one per logical group (components, models, tasks, tests).

### Phase 3: Remove model-specific fields from ArchitectureConfig

Once each model family owns its fields, remove them from
`ArchitectureConfig`. The dataclass shrinks from ~170 to ~55 fields.

### What does a half-migrated codebase look like?

During migration, some models use new-style per-model configs while
others still use the old monolith. This is safe because `_config_from_hf()`
already supports both paths:

```python
# _config_from_hf resolution (UNCHANGED):
# 1. module_class.config_class → use it (new-style)
# 2. Registry config_class → use it (new-style)
# 3. Fall back to ArchitectureConfig.from_transformers() (old monolith)
```

**Concrete example of half-migrated state:**

```
MIGRATED (new-style):
  Falcon → FalconConfig.from_transformers() in models/falcon.py
  Whisper → WhisperConfig.from_transformers() in _configs.py (already done)
  Phi4MM → Phi4MMConfig.from_transformers() in models/phi.py

STILL ON OLD PATH:
  Llama → ArchitectureConfig.from_transformers() in _configs.py (shared core)
  Qwen2 → ArchitectureConfig.from_transformers() in _configs.py (shared core)
  Gemma → ArchitectureConfig.from_transformers() in _configs.py (needs Gemma2Config)
```

The old monolith shrinks with each migration, but it always works for
unmigrated models. There is never a broken intermediate state.

**Registry entries don't change during migration.** The `config_class`
attribute on the module class is sufficient — `_config_from_hf()` reads
it automatically. No need to update 267 registry entries.

**Tests verify both paths.** Existing `build_graph_test.py` parametrized
tests continue to work regardless of which config path a model uses.
The output (ONNX graph) is identical — only the config construction
path changes.

### What does NOT change

- `BaseModelConfig` — stays as-is
- Component constructor signatures — components accept `ArchitectureConfig`
  which is now a proper base class. Subclasses are passed polymorphically.
- `ModelPackage` — unchanged
- `ModelTask` — unchanged
- `build()` and `build_from_module()` — unchanged (they call
  `_config_from_hf` which dispatches to the right config class)

### Validation Policy

**During migration (Phases 0-2):** `validate()` stays on
`ArchitectureConfig` ONLY. Subclasses MUST NOT override `validate()`
during migration phases. Reason: if a Phase 1 subclass overrides
`validate()` to check its own fields, it creates coupling between
config migration and validation logic that makes rollback harder.

**After migration (Phase 3):** Once the migration is stable and all
model families have settled into their config subclasses, subclasses
MAY add their own `validate()` override (calling `super().validate()`
first).

**DEFAULT_INT sentinel propagation:** When `ArchitectureConfig.from_transformers()`
returns a config with `DEFAULT_INT=-42` sentinel values (e.g., for
encoder-decoder fields that don't apply to a decoder-only model),
these propagate through `_shallow_fields(base)` into the subclass
instance. This is correct — the sentinels mean "not set" and the
base `validate()` only flags them for fields that are actually
required. Encoder model subclasses that declare additional validated
fields must explicitly set those fields in their `from_transformers()`.

### Testing migrated configs

Each migration PR includes a **round-trip test** that verifies the new
config path produces the exact same config as the old path:

```python
def test_falcon_config_migration():
    """Verify FalconConfig produces identical results to old path."""
    hf_config = AutoConfig.from_pretrained("tiiuae/falcon-7b")

    # New path: FalconConfig.from_transformers()
    new_config = FalconConfig.from_transformers(hf_config)

    # Old path: ArchitectureConfig.from_transformers() (before removal)
    old_config = ArchitectureConfig.from_transformers(hf_config)

    # All shared fields must match
    for f in dataclasses.fields(ArchitectureConfig):
        assert getattr(new_config, f.name) == getattr(old_config, f.name), \
            f"Field {f.name} differs: {getattr(new_config, f.name)} vs {getattr(old_config, f.name)}"
```

Additionally, the existing `build_graph_test.py` parametrized tests
serve as integration tests — if the migrated config produces a
different ONNX graph, the test will fail.

### preprocess_weights() interaction with config subclasses

`preprocess_weights()` is a method on the MODEL class, not the config
class. It receives the config as an argument:

```python
class FalconCausalLMModel(CausalLMModel):
    config_class = FalconConfig  # NEW: points to subclass

    def preprocess_weights(self, config, weights):
        # config is now FalconConfig (subclass of ArchitectureConfig)
        # This works because FalconConfig inherits all shared fields
        if config.alibi:  # FalconConfig-specific field
            ...
```

**No changes needed** to `preprocess_weights()` during migration:
- Subclass configs are `isinstance(config, ArchitectureConfig)` → True
- All shared fields are inherited, so existing field access works
- Model-specific fields (e.g., `config.alibi`) that were previously
  on ArchitectureConfig are now on the subclass — same attribute name,
  same access pattern

---

## 4. How the Registry Entry Changes

**Current** (from `_registry.py`):

```python
reg.register("whisper", WhisperForConditionalGeneration,
             task="speech-to-text", config_class=WhisperConfig)
```

Most models don't specify `config_class` — the registry defaults to
`None`, and `_config_from_hf` falls through to `ArchitectureConfig`.

**New pattern:** Model classes declare `config_class` as a class attribute
(already supported by `_config_from_hf`):

```python
class FalconCausalLMModel(CausalLMModel):
    config_class = FalconConfig  # ← resolution happens automatically
```

OR register in the registry (for models that share a module class but
need different configs):

```python
reg.register("bloom", FalconCausalLMModel, config_class=FalconConfig)
reg.register("falcon", FalconCausalLMModel, config_class=FalconConfig)
```

**No changes needed to `_config_from_hf()`** — it already checks
`module_class.config_class` first, then registry config_class, then
falls back. The resolution order is correct.

---

## 5. Code Examples

### Example A: Simple model (Llama-compatible)

Llama, Mistral, Qwen2, and ~40 other architectures use `CausalLMModel`
with `ArchitectureConfig` directly. **Nothing changes for them.**

```python
# models/base.py — UNCHANGED
class CausalLMModel(nn.Module):
    config_class: type = ArchitectureConfig  # uses shared config
    default_task: str = "text-generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
```

```python
# _registry.py — UNCHANGED
reg.register("llama", CausalLMModel)      # uses ArchitectureConfig by default
reg.register("mistral", CausalLMModel)
reg.register("qwen2", CausalLMModel)
```

### Example B: Complex model (Phi4-MM multimodal)

```python
# models/phi.py — NEW config class added to existing file

@dataclasses.dataclass
class Phi4MMConfig(VisionLanguageConfig):
    """Configuration for Phi4-MM multimodal (vision + audio)."""
    audio: AudioConfig | None = None
    vision_lora: dict | None = None
    speech_lora: dict | None = None
    image_crop_size: int | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None, **kw):
        # Get base VL config (handles text + vision extraction)
        base = VisionLanguageConfig.from_transformers(
            config, parent_config=parent_config, **kw
        )

        # Phi4MM: override with hardcoded SigLIP vision encoder params
        embd_layer = getattr(config, "embd_layer", None)
        crop_size = None
        if isinstance(embd_layer, dict):
            crop_size = embd_layer.get("image_embd_layer", {}).get("crop_size")

        vision = VisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=crop_size or 448,
            patch_size=14,
            norm_eps=1e-6,
            image_token_id=getattr(config, "special_image_token_id", 200010),
            image_crop_size=crop_size,
        )

        # Audio config from audio_processor dict
        audio = None
        audio_processor = getattr(config, "audio_processor", None)
        if isinstance(audio_processor, dict) and "config" in audio_processor:
            ac = audio_processor["config"]
            nemo = ac.get("nemo_conv_settings", {})
            rel_bias = ac.get("relative_attention_bias_args", {})
            audio = AudioConfig(
                attention_dim=ac.get("attention_dim"),
                attention_heads=ac.get("attention_heads"),
                num_blocks=ac.get("num_blocks"),
                linear_units=ac.get("linear_units"),
                kernel_size=ac.get("kernel_size"),
                input_size=ac.get("input_size"),
                conv_channels=nemo.get("conv_channels"),
                t5_bias_max_distance=rel_bias.get("t5_bias_max_distance"),
                projection_hidden_size=config.hidden_size,
            )
            audio_config_dict = getattr(config, "audio_config", None)
            if audio_config_dict is not None:
                ad = audio_config_dict if isinstance(audio_config_dict, dict) else vars(audio_config_dict)
                audio.token_id = ad.get("audio_token_id")

        return cls(
            **_shallow_fields(base),
            vision=vision,
            audio=audio,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
            image_crop_size=crop_size,
        )


class Phi4MMMultiModalModel(nn.Module):
    config_class = Phi4MMConfig  # ← auto-dispatched by _config_from_hf
    default_task = "multimodal"
    # ... rest unchanged
```

### Example C: Adding a brand new model (contributor experience)

A contributor adding "NewArch" that's Llama-compatible with one extra
field:

```python
# models/newarch.py — the ONLY file they create

from __future__ import annotations

import dataclasses
from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel


@dataclasses.dataclass
class NewArchConfig(ArchitectureConfig):
    """NewArch adds a custom gating parameter."""
    gate_scale: float = 1.0

    @classmethod
    def from_transformers(cls, config, **kw) -> NewArchConfig:
        base = ArchitectureConfig.from_transformers(
            config, **kw
        )
        return cls(
            **_shallow_fields(base),
            gate_scale=getattr(config, "gate_scale", 1.0),
        )


class NewArchCausalLMModel(CausalLMModel):
    config_class = NewArchConfig
    # ... override forward() or preprocess_weights() as needed
```

Then register (the only other file touched):

```python
# _registry.py
reg.register("newarch", NewArchCausalLMModel)
# No config_class needed in register() — it reads from the class attribute
```

**Zero changes to `_configs.py`.** ✅

---

## 6. What Happens to Flat `vision_*` Fields vs Nested `VisionConfig`

### Current state

Both exist. `from_transformers()` populates both (lines 752-778 copy
nested → flat). Components and models read the flat fields. The nested
`VisionConfig` has zero model-layer consumers today.

### New design

**`VisionConfig` becomes the canonical source.** Models that need
vision config use `VisionLanguageConfig` which has a `vision: VisionConfig`
field.

**Flat `vision_*` fields are removed from `ArchitectureConfig`** in
Phase 3. Components that currently read `config.vision_hidden_size`
are updated to read `config.vision.hidden_size`.

**Affected component files** (from exploration):
- `components/_vision.py` — reads `vision_hidden_size`, `vision_num_*`, etc.
- `components/_qwen25_vl_vision.py` — reads various vision_* fields
- `components/_qwen3_vl_vision.py` — same

These are already model-specific components (they're named after Qwen!),
so updating them to read from `config.vision.*` is natural.

**Affected task files:**
- `tasks/_vision_language_3model.py` — reads `vision_patch_size`, `temporal_patch_size`, etc.
- `tasks/_vision_language.py` — same
- `tasks/_multimodal.py` — reads `audio_input_size`

### Migration path for flat fields

```python
# Phase 2: Add property aliases on ArchitectureConfig for backward compat
@property
def vision_hidden_size(self) -> int | None:
    return self.vision.hidden_size if self.vision else None

# Phase 3: Remove properties, update all consumers to use config.vision.*
```

---

## Appendix: Complete Field Inventory

### Fields staying on ArchitectureConfig (~55-60)

**Note:** The original estimate of ~35 was incorrect. Counting all
categories (core + attention + MoE + MLA + encoder-decoder + etc.)
yields ~55-60 fields on the base class. This is still a significant
reduction from ~170, and the improvement is real: the removed fields
are the model-specific ones that caused confusion (Granite multipliers,
Gemma2 softcapping, SegFormer fields, etc.).

**Core (from BaseModelConfig):** vocab_size, hidden_size,
intermediate_size, num_hidden_layers, num_attention_heads,
num_key_value_heads, head_dim, hidden_act, pad_token_id,
tie_word_embeddings, attn_qkv_bias, attn_o_bias, dtype

**Attention:** layer_types, full_attention_interval, sliding_window,
attn_output_gate, attn_qk_norm, attn_qk_norm_full

**Linear attention (DeltaNet):** linear_conv_kernel_dim,
linear_key_head_dim, linear_value_head_dim, linear_num_key_heads,
linear_num_value_heads

**Norm/MLP:** rms_norm_eps, mlp_bias, max_position_embeddings

**RoPE:** rope_type, rope_theta, rope_scaling, partial_rotary_factor,
rope_local_base_freq, original_max_position_embeddings, rope_interleave

**MRoPE:** mrope_section, mrope_interleaved

**Encoder-only:** type_vocab_size

**Encoder-decoder:** num_decoder_layers, relative_attention_num_buckets,
relative_attention_max_distance

**Standalone vision:** image_size, patch_size, num_channels

**MoE (shared):** num_local_experts, num_experts_per_tok,
moe_intermediate_size, shared_expert_intermediate_size, norm_topk_prob,
decoder_sparse_step, n_group, topk_group, routed_scaling_factor,
scoring_func, topk_method, first_k_dense_replace, n_shared_experts

**MLA (shared):** q_lora_rank, kv_lora_rank, qk_nope_head_dim,
qk_rope_head_dim, v_head_dim

**Nested sub-configs:** vision (VisionConfig), audio (AudioConfig),
tts (TTSConfig), codec_decoder, codec_encoder

### Fields moving to model-specific configs

| Field(s) | Current location | New location |
|---|---|---|
| alibi, parallel_attn | ArchitectureConfig | FalconConfig |
| embedding_multiplier, attention_multiplier, logits_scaling, residual_multiplier | ArchitectureConfig | GraniteConfig |
| attn_logit_softcapping, final_logit_softcapping, query_pre_attn_scalar | ArchitectureConfig | Gemma2Config |
| altup_*, laurel_*, hidden_size_per_layer_input, vocab_size_per_layer_input | ArchitectureConfig | Gemma3nConfig |
| cross_attention_layers | ArchitectureConfig | MllamaConfig |
| num_detection_tokens, num_labels | ArchitectureConfig | YolosConfig |
| neck_hidden_sizes, reassemble_factors, fusion_hidden_size, head_hidden_size, backbone_out_indices | ArchitectureConfig | DepthAnythingConfig |
| segformer_* (8 fields), decoder_hidden_size | ArchitectureConfig | SegformerConfig |
| sam2_* (5 fields) | ArchitectureConfig | Sam2Config |
| vision_* flat fields (20 fields) | ArchitectureConfig | Removed (use VisionConfig) |
| audio_* flat fields (10 fields) | ArchitectureConfig | Removed (use AudioConfig) |
| image_crop_size, vision_lora, speech_lora | ArchitectureConfig | Phi4MMConfig |

### Fields that stay as nested sub-configs on ArchitectureConfig

VisionConfig, AudioConfig, TTSConfig, CodecDecoderConfig,
CodecEncoderConfig — these remain as optional nested fields on
ArchitectureConfig. Model-specific subclasses (VisionLanguageConfig,
Phi4MMConfig) make them required/have them always populated.

---

## Summary

| Metric | Current | After redesign |
|---|---|---|
| ArchitectureConfig fields | ~170 | ~55 |
| from_transformers() lines | 573 | ~120 (shared core) |
| Files touched to add simple model | 2 (registry + configs) | 1 (registry) |
| Files touched to add complex model | 2 (registry + configs) | 2 (model file + registry) |
| Model-specific if/elif branches | 5 in central monolith | 0 (distributed to model configs) |
| Config classes with own from_transformers | 1 (Whisper) | ~10 (one per model family) |
