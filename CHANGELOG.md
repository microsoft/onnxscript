# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Wave 9

#### Added

- `PostGatedRMSNorm` component for Qwen3.5 DeltaNet — gate-after-norm
  variant (`RMSNorm(x) * SiLU(gate)`) separate from Mamba2's
  gate-before-norm `GatedRMSNorm`.
- `merge_lora_weights()` in weight utils — merges PEFT LoRA adapters
  (`*.lora_A.weight` / `*.lora_B.weight`) into base weights at load time.
- Vision encoder integration tests (ViT + CLIP) with HF parity.
- KeyError guard for missing qweight in GPTQ/AWQ preprocessors —
  raises `ValueError` with context instead of raw `KeyError`.
- GGUF support proposal document (`docs/design/gguf-support-proposal.md`).

#### Fixed

- AWQ zero-point offset: per-nibble subtraction for 4-bit quantization.
  Byte-level `0x88 - 1 = 0x87` was wrong; now unpacks nibbles first
  to get correct `0x77`.
- CLIP class embedding `Unsqueeze` axes `[0,0]` → `[0,1]`.

### Wave 10 — GGUF Import

#### Added

- GGUF import pipeline (`build_from_gguf()`) — converts `.gguf` model
  files to ONNX via the standard build pipeline. Phase 1 dequantizes
  all tensors to float; Phase 2 will preserve quantization.
- `GGUFModel` reader wrapping `gguf.GGUFReader` with typed metadata
  parsing, lazy tensor iteration, and O(1) tensor lookup.
- `gguf_to_config()` mapping GGUF metadata → `ArchitectureConfig` with
  architecture-specific key resolution (HF `GGUF_CONFIG_MAPPING`
  fallback to standard GGUF keys).
- GGUF → HF tensor name mapping for 8 architecture families (Llama,
  Gemma, Phi3, Falcon, GPT-2, Mamba, MoE variants) with `{bid}`
  block-index expansion.
- Architecture-specific tensor processors: Llama Q/K reverse-permute,
  Gemma/Nemotron norm offset (+1), GPT-2 weight transpose, Mamba
  conv1d unsqueeze + A_log transform.
- `build-gguf` CLI subcommand with `--output`, `--dtype`,
  `--external-data`, and `--keep-quantized` (Phase 2 placeholder) flags.
- `is_known_skip()` for differentiated GGUF tensor logging — separates
  intentionally skipped tensors (tokenizer, rope_freqs) from
  genuinely unmapped ones.
- `gguf` optional dependency group in `pyproject.toml`
  (`pip install mobius-ai[gguf]`).
- 28 unit tests for GGUF reader, config mapping, tensor mapping,
  tensor processors, and CLI using synthetic GGUF files.

#### Changed

- GGUF tensor mapping cached with `lru_cache` for performance —
  `_build_mapping()` called once per architecture instead of per tensor.
- `_reverse_permute` simplified to single `n_head` parameter.
- `_dequantize_tensor` helper extracted to eliminate duplication between
  `tensor_items()` and `get_tensor()` in reader.

#### Fixed

- `gguf_to_config()` raises `ValueError` for missing critical metadata
  fields (`embedding_length`, `block_count`) instead of silently
  defaulting to hardcoded values.

#### Documentation

- GGUF support proposal (`docs/design/gguf-support-proposal.md`) with
  quantization type catalog, QDQ vs MatMulNBits analysis, and phased
  implementation plan.

### Wave 10 — Other

#### Added

- T5 variant support: gated FFN activation (`T5_GATED_ACT_TO_HF`),
  `scale_decoder_outputs` config field, and integration tests for
  mT5/FLAN-T5/UL2.

#### Changed

- Config Phase 2a: wrapped all config subclass `__init__` methods to
  accept flat `vision_*` kwargs for backward compatibility with nested
  `VisionConfig`.
- Skip `hidden_size / num_attention_heads` divisibility check when
  `head_dim` is explicitly provided in config.

### Sprint 8 Highlights

- **FusedMatMul rewrite rule**: New `Transpose + MatMul → FusedMatMul(transB=1)`
  rule eliminates 197 Transpose nodes per LLM model (every Linear layer).
  CLI `--optimize=all` now registers all 6 rewrite rules (was only 3).
- **Quantization integration tests**: GPTQ and AWQ end-to-end tests with
  synthetic weights verify full pipeline (build → preprocess → apply → ORT
  inference). Found and fixed `_reshape_packed_qzeros` overestimate bug.
- **Mamba2 HF parity**: Integration test with step-by-step numerical
  comparison against HuggingFace. Fixed `GatedRMSNorm` gate ordering
  (SiLU(gate) before normalization).
- **Weight loading fixes**: Consolidated shape mismatch logic into
  `_assign_weight()`, removed dead symbolic-dim guard, fixed Jamba weight
  alignment, cleaned up 48 stale xfails.
- **SSM/T5 correctness**: Bamba/Mamba substring match → `endswith()`,
  T5 logit scaling guard for `tie_word_embeddings`, Mamba2 `head_dim`
  divisibility validation.

### Added

- `FusedMatMul` rewrite rule: fuses `Transpose(weight, [1,0]) + MatMul`
  into `com.microsoft::FusedMatMul(transB=1)` — eliminates one node per
  linear projection (197 fusions in Qwen3-0.6B).
- GPTQ/AWQ end-to-end integration tests with synthetic weights —
  build tiny quantized Llama, preprocess weights, run ORT inference.
- AWQ zero-point offset verification integration test.
- Mamba2 integration test with step-by-step HF parity comparison
  (GatedRMSNorm, Mamba2Block, full model logits).
- All 6 rewrite rules registered in CLI `--optimize` rule map
  (`bias_gelu`, `fused_matmul`, `group_query_attention`,
  `packed_attention`, `skip_layer_norm`, `skip_norm`).

### Fixed

- `GatedRMSNorm` gate ordering: apply `SiLU(gate)` before normalization,
  matching HuggingFace `MambaRMSNormGated`.
- SSM weight rename: greedy substring match (`if param in key`) replaced
  with `key.endswith(param)` in Bamba/Mamba to prevent `.mamba.D` from
  matching `.mamba.Dropout`.
- T5 logit scaling: `hidden_size**-0.5` multiply now guarded by
  `tie_word_embeddings` flag, matching HuggingFace T5 behavior.
- Mamba2 `head_dim` divisibility: `from_transformers()` now raises
  `ValueError` when `d_inner % num_heads != 0`.
- Jamba weight alignment: removed incorrect `model.` prefix stripping
  in `preprocess_weights()`.
- Weight shape mismatch handling consolidated into single
  `_assign_weight()` helper in `_weight_loading.py`.
- Dead symbolic-dim guard removed from `_assign_weight()` — ONNX
  initializers always have concrete integer shapes.
- 48 stale weight alignment test xfails cleaned up (37 genuine remain).
- Mamba2 cache defaults: replaced misleading Bamba-9B-specific values
  (128/64/256) with 0 — configs must provide real values.
- `_reshape_packed_qzeros` overestimated output size when
  `n_groups * bits < 32` — fixed by deriving `n_blocks` from
  `qweight` shape.

### Changed

- Rewrite rules included in default test command (removed
  `--ignore=src/mobius/rewrite_rules`).
- Stale symbolic dim docstring cleaned up in weight loading module.

### Sprint 7 Highlights

- **Quantization support**: `QuantizedLinear` component with `MatMulNBits`,
  `QuantizationConfig`, GPTQ weight preprocessing, and `linear_class`
  injection into all decoder layer projections via `quantized_linear_factory`.
- **New models**: `BambaCausalLMModel` (Mamba2/SSD + Attention hybrid),
  `InternVL2Model` (dedicated VL model with InternViT + pixel shuffle).
- **Auto-export pipeline**: `auto_export()` chains build → apply weights →
  genai_config.json → save → tokenizer copy for ORT-GenAI deployment.
- **Documentation**: Model catalog expanded 26→272 pages via fixed
  `_generate_models.py`; added `docs` optional dependency group.
- **Error diagnostics**: Registry fuzzy matching with `difflib`, `hidden_act`
  guard, weight shape mismatch warnings.
- **Rewrite rules**: SkipLayerNorm bias-free variant for models without LN
  bias; BiasGelu approximate attribute guard for exact Gelu.
- **Test coverage**: 154 new component tests, quantization integration tests,
  generation loop tests, task I/O contract tests.

### Added

- Auto-fallback registry for unregistered model types — when
  `build()` encounters an unknown `model_type`, heuristically detects
  Llama-like or MoE architectures and routes to the appropriate model
  class. Logs at INFO level when using fallback.
- `SelectiveScan` and `MambaBlock` SSM components for Selective State
  Space Models. `SelectiveScan` implements core S6 recurrence;
  `MambaBlock` composes Conv1D + SSM + projections.
- `integration-fast` CI job running fast integration tests with
  HuggingFace model cache.
- Registered DiT (PixArt) and VideoVAE (CogVideoX) in diffusers
  class map for pipeline builder support.
- `HunyuanDiT2DModel` diffusion transformer with AdaLN-Shift, QK-norm,
  GEGLU FFN, and U-Net-style skip connections.
- `QFormer` component for BLIP-2 style VLMs with learned query tokens
  and cross-attention to visual features.
- `_weight_utils.py` module with shared `split_fused_qkv`,
  `split_gate_up_proj`, and `strip_prefix` helpers for weight
  preprocessing.
- Type annotations (`builder.OpBuilder`, `ir.Value`) to `forward()`
  methods across 13 model files (diffusion, vision, language, speech).
- 10 additional config helper tests for Gemma3 nested rope and legacy
  `rope_type` key.
- Unmapped weight warnings in `ModelPackage.apply_weights()` — logs at
  INFO level for weights not applied (may be tied or unused).
- 21 unit tests for `_diffusers_builder.py` (config resolution, error
  handling, weight loading).
- `pytest-xdist` for parallel test execution (`pytest -n auto`).
- Reference examples table in adding-a-new-model skill documentation.
- 23 unit tests for Seq2Seq, Denoising, and VAE tasks (I/O contracts,
  KV cache naming, input dtypes, ModelPackage structure).
- `MambaCausalLMModel` and `SSMCausalLMTask` for Mamba selective state
  space models — embedding → N×(MambaBlock + RMSNorm) → logits with
  conv_state + ssm_state carry (no attention mask or KV cache).
- Integration tests for Whisper encoder-decoder (`openai/whisper-tiny`)
  and Gemma3 multimodal 3-model split (tiny config, no download).
  Added `whisper-tiny` and `test_gemma3_multimodal` to CI `-k` filter.
- `Blip2Model` vision-language model with Q-Former bridge — ViT encoder
  → Q-Former cross-attention → language model, using 3-model split
  (decoder, vision encoder, embedding).
- Qwen3.5-VL integration tests: random-weight 3-model VL pipeline test
  and DeltaNet state carry test verifying conv_state/recurrent_state
  update across consecutive decode steps.
- End-to-end generation loop tests for top 5 CausalLM architectures
  (Llama, Qwen2, Phi3, Gemma2, Mistral) — random-weight models with
  5-step autoregressive decode verifying KV cache growth and finite
  logits.
- Graph construction benchmarks for 10 representative models with
  regression guard (`MAX_BUILD_TIME_SECONDS` threshold).
- `JambaCausalLMModel` hybrid SSM+Attention model with MoE support —
  interleaved Mamba SSM and Transformer attention layers with optional
  Mixture-of-Experts FFN, per-layer conv_state/ssm_state carry, and
  dt/B/C LayerNorm in SSM projections.
- `CogVideoX3DTransformer2DModel` 3D video diffusion transformer with
  temporal attention, 3D positional embeddings, and expert-block
  adaptive LayerNorm.
- FalconMamba registration as alias for `MambaCausalLMModel`.
- `SkipLayerNormalization` rewrite rule: fuses Add + LayerNormalization
  into `com.microsoft::SkipLayerNormalization` for GPT-2/BERT-style
  models (24/25 fusions in GPT-2).
- `BiasGelu` rewrite rule: fuses Add + Gelu into
  `com.microsoft::BiasGelu` for FFN bias+activation (12 fusions in
  GPT-2). Includes `approximate` attribute guard to skip exact Gelu.
- `SkipLayerNormalization` bias-free variant: matches LayerNorm with
  2 inputs (no bias) in addition to the 3-input pattern.
- ORT-GenAI integration module (`integrations/ort_genai/`) with
  `GenaiConfigGenerator` for genai_config.json generation.
- `scaffold` CLI command for generating new model boilerplate files
  with templates for all base types (causal-lm, encoder-decoder,
  vision-encoder, diffusion).
- BLIP-2 VLM integration test (4 tests: structure, vision, embedding,
  decoder with random weights).
- BART/T5 seq2seq integration tests with encoder-decoder verification.
- VisionConfig bidirectional sync unit tests (4 tests guarding the
  flat ↔ nested __post_init__ invariant).
- Qwen3.5-VL HF parity integration tests (random-weight 3-model VL
  pipeline and DeltaNet state carry verification).
- End-to-end generation loop tests for 5 CausalLM architectures
  (Llama, Qwen2, Phi3, Gemma2, Mistral) with autoregressive decode.
- ORT-GenAI auto-export pipeline (`auto_export()`) chaining build →
  apply weights → genai_config.json → save → tokenizer copy.
- `InternVL2Model` dedicated VL model with InternViT encoder, pixel
  shuffle downsampling, and 2-layer MLP projector — replaces incorrect
  LLaVA mapping.
- `QuantizedLinear` component using `MatMulNBits` (com.microsoft) for
  INT4/INT8 weight-only quantized models (GPTQ, AWQ).
- `QuantizationConfig` dataclass with `from_transformers()` factory for
  reading HF `quantization_config`.
- Quantization Phase 2: model integration with `quantized_linear_factory`
  closure and GPTQ weight preprocessing.
- `BambaCausalLMModel` hybrid Mamba2/SSD + Attention model with
  interleaved SSM and transformer layers.
- `Mamba2Scan` and `Mamba2Block` components for Mamba-2 SSD (Structured
  State Space Duality) architecture.
- Registry fuzzy matching: unknown `model_type` now suggests closest
  matches via `difflib.get_close_matches` instead of dumping all 271+
  registered types.
- 154 unit tests for 10 previously untested components (Attention, MLP,
  DecoderLayer, Encoder, Conv, LoRA, Whisper, GatedDeltaNet, Scan
  utilities).
- Sphinx documentation site with auto-generated model catalog (272
  model pages from registry metadata). Added `docs` optional dependency
  group to `pyproject.toml`.
- `hidden_act=None` guard in `get_activation()` with descriptive error
  message listing valid activation functions.
- Weight shape mismatch warnings in `apply_weights()` — logs
  expected vs actual shape for debugging.

### Removed

- Dead `_rename_gpt2_weight()` function in `gpt2.py` — unreferenced
  since weight name alignment refactor.

- `SUPPORTED_ARCHITECTURES` allowlist in `_configs.py` — was blocking
  42–66 model types that were registered but not in the allowlist.

### Changed

- Refactored `falcon.py`, `phi3.py`, `phi.py` to use shared weight
  utils instead of inline QKV splitting.
- Aligned module attribute names to HuggingFace weight conventions for
  Falcon, GPT-2, ModernBERT, InternLM, and BERT, eliminating 30+
  `preprocess_weights` renames.
- Added module docstrings and HF class references to top 10 model files
  (Llama, Qwen2, Phi-3, Gemma, Mistral, DeepSeek, BERT, GPT-2, Falcon,
  Phi).
- Config Phase 0a: extracted `_extract_rope_config`,
  `_extract_vision_config`, `_extract_audio_config` helpers from
  `from_transformers()` monolith.
- Unmapped weights logged at INFO level (not WARNING) to reduce noise
  for models with tied embeddings.
- Enhanced 4 TODO comments with category tags and implementation context.
- Fixed TTS task layer violation: removed model import, use direct
  config access.
- Extracted shared diffusion components (`AdaLayerNormZero`,
  `TimestepEmbedding`, `PatchEmbed`, `DiffusionFFN`) from `dit.py` into
  `components/_diffusion.py`; updated `dit.py`, `flux_sd3.py`, and
  `hunyuan_dit.py` to use shared imports.
- Replaced bare `except Exception` with specific exception types
  (`OSError`, `ValueError`, `json.JSONDecodeError`) and added DEBUG-level
  logging in `_config_resolver.py`, `_diffusers_builder.py`, and
  `__main__.py`.
- Added dependency lower bounds: `onnxscript>=0.6.0`, `onnx_ir>=0.1.0`,
  `numpy>=1.24.0`, `torch>=2.1.0`.
- Lazy-import heavy dependencies (`torch`, `transformers`,
  `safetensors.torch`) in CLI for faster `list`/`info` subcommands.
- Mllama cross-attention K/V now cached after first computation in
  `MllamaVisionLanguageTask` decoder, avoiding redundant recomputation
  during decode steps.
- Config Phase 2: migrated all flat `vision_*` fields to nested
  `config.vision.*` accessors across production code (models and tasks).
  Backward compatibility maintained via `__post_init__` bidirectional
  sync.

### Fixed

- Tautological assertion in diffusers builder tests (always-True →
  meaningful `assert_called_once`).
- Temp file leak in `examples/diffusion.py` VAE test — replaced
  `NamedTemporaryFile(delete=False)` with `mkdtemp` + `try/finally`
  cleanup.
- RWKV SSM models now rejected in auto-fallback registry instead of
  silently producing incorrect graphs.
- Mamba SSM weight path mapping: `preprocess_weights()` now correctly
  maps flat HF mixer params (`.mixer.A_log`, `.mixer.D`, etc.) to
  nested ONNX params (`.mixer.ssm.A_log`, `.mixer.ssm.D`). Without
  this fix, all SSM params were silently dropped during weight loading.
- `MambaConfig.from_transformers()` handles `intermediate_size=0`
  (falls back to `hidden_size * expand`) and `time_step_rank="auto"`
  (resolves to `ceil(hidden_size / 16)`).
- bfloat16 conversion in test `_fill_random_weights`: uses upper-16-bit
  truncation of float32 instead of broken `.view(np.uint16)` that
  doubled the last dimension.
- Python keyword validation in `scaffold` CLI: rejects reserved words
  like `for`, `class`, `import` that pass regex but create unimportable
  module files.
- TOCTOU race in scaffold file writing: replaced `os.path.exists()` +
  `open("w")` with atomic `open("x")`.
- ORT-GenAI VLM decoder input detection and required `image_token_id`
  field.
- T5 encoder weight mapping: `layer.1.layer_norm` → `ffn_norm`.
- `JambaConfig` duplicate `num_experts` field removed.
- Cross-attention KV cache recomputation in encoder-decoder models:
  `EncoderDecoderAttention` and `WhisperAttention` now properly cache
  cross-attention K/V from the first decode step instead of
  recomputing every step.
- `BiasGelu` rewrite rule now only matches `Gelu(approximate='tanh')`;
  exact Gelu (`approximate='none'`) is no longer incorrectly fused.
- `QuantizedLinear` rejects `block_size < 16` per ORT MatMulNBits spec
  (was only checking positive power-of-2).
- `SkipLayerNormalization` rewrite rule handles bias-free
  LayerNormalization nodes (2-input pattern).
- `auto_export` no longer excludes valid `image_token_id=0`; VLM
  detection tightened to require both `vision` and `embedding` keys.
- `auto_export` guards `pkg.config` access with descriptive error for
  unsupported diffusion models.
- `InternVL2Model` raises `ValueError` when `image_token_id` is None
  instead of silently defaulting to 0.
- Sphinx model catalog generator (`_generate_models.py`) produces pages
  for all 272 registered model types (was only generating 26 due to
  `ModelRegistration` API change).

## [0.1.0] - 2026-02-27

### Added

- Declarative ONNX model construction using `onnxscript.nn` — builds graphs
  directly without tracing or exporting PyTorch.
- Support for 267 registered model types across ~55 architecture families:
  text generation (Llama, Mistral, Qwen, Phi, Gemma, …), MoE (Mixtral,
  DeepSeek, DBRX, …), multimodal (Gemma 3, LLaVA, Phi-4MM, Qwen-VL, …),
  encoder-only (BERT, RoBERTa, DeBERTa, …), encoder-decoder (BART, T5,
  Whisper, …), vision (ViT, CLIP, SigLIP, …), audio (Wav2Vec2, HuBERT, …),
  and diffusion (Stable Diffusion, Flux, SD3, DiT).
- 14 task types: CausalLM, VisionLanguage, Seq2Seq, FeatureExtraction,
  ImageClassification, SpeechToText, AudioFeatureExtraction, Denoising,
  VAE, ControlNet, Adapter, MultiModal, ObjectDetection, and Codec.
- 56+ reusable components (Attention, MLP, RMSNorm, RoPE, MoELayer,
  VisionEncoder, MultiModalProjectors, …).
- CLI (`mobius build/list/info`) for building and exporting models.
- Python API: `build()` for HuggingFace model IDs, `build_from_module()` for
  custom modules, `ModelPackage` for multi-component model management.
- HuggingFace weight loading via safetensors (no pickle deserialization).
- Automatic dtype detection and casting (float32, float16, bfloat16) with
  `ir.LazyTensor` for memory-efficient dtype conversion.
- ONNX graph rewrite rules for GroupQueryAttention, PackedAttention, and
  SkipNorm fusion.
- 10 examples covering text generation, multimodal, ASR, TTS, and
  ORT-GenAI integration.
- 11 contribution skills (`.github/skills/`) for AI-agent-assisted
  development: adding models, writing tests, debugging VL pipelines,
  rewrite rules, and more.
