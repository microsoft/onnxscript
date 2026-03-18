# Multi-Tier Testing Strategy

**Status:** Design proposal — ready for review
**Date:** 2026-03-10

---

## Executive Summary

This document defines the multi-tier testing strategy for
mobius, a package that declaratively constructs ONNX models
for 270+ generative AI architectures.  The strategy addresses three
challenges unique to this project:

1. **Breadth**: 270+ registered model types, each a distinct graph.
2. **Layered correctness**: A model can build without errors yet produce
   wrong numerics.  Graph validity, weight alignment, numerical parity,
   and generation quality are separate failure modes.
3. **Cost asymmetry**: Graph construction is instant; downloading real
   weights and running inference takes minutes per model.

The strategy uses a **test pyramid** with six tiers (L0–L5), where
cheap broad tests catch most bugs and expensive narrow tests provide
production confidence.  Each tier answers a specific question, has its
own CI cadence, and covers a distinct set of failure modes.

### Key design principle

> Test all variants at the cheapest tier; test one representative at
> the most expensive tier.

### Related documents

- [Test Infrastructure Map](test-infrastructure-map.md) — inventory of
  all test files, markers, fixtures, and CI workflows
- [Golden File Infrastructure Design](golden-infra-design.md) — L4/L5
  golden file format, YAML schema, and `generate_golden.py` CLI
- [Performance Benchmarking](perf-benchmarking.md) — build-time and
  model-size regression detection
- [Testing Strategy Analysis](../research/testing-strategy-analysis.md)
  — research analysis with product-manager and radical-thinker
  perspectives

---

## 1. Tier Definitions

### Overview

```
            Coverage (model types)
  L0  ████████████████████████████  270+     instant    Registry exists
  L1  ████████████████████████████  270+     <1s each   Graph builds
  L2  ████████████████████████████  270+     ~5s each   Real HF config validates
  L3  ████████████████████████████  270+     3-10s each Synthetic parity (random weights)
  L4  ██████████                    ~50      minutes    Checkpoint-verified (real weights)
  L5  █████                         ~20      minutes    Generation E2E (token match)
```

### L0 — Registered

| Aspect | Detail |
|--------|--------|
| **Question** | Does this `model_type` exist in the registry? |
| **What runs** | Registry lookup; no graph construction |
| **Speed** | Instant |
| **Coverage** | All 270+ registered types |
| **Failure mode** | Missing registration, wrong task mapping |
| **File** | Implicit — `_registry.py` entries; verified by L1 parametrization |

L0 is implicit: if a `model_type` string is registered, it appears on the
dashboard.  No separate test file is needed — L0 is trivially covered
by L1 parametrization (if L1 lists the model_type, registration is
implicitly proven).

### L1 — Graph Construction (Smoke)

| Aspect | Detail |
|--------|--------|
| **Question** | Does the ONNX graph build without errors? |
| **What runs** | Build graph with tiny config (hidden=64, 2 layers, 256 vocab); verify I/O names, initializer existence, op types |
| **Speed** | <1s per model |
| **Coverage** | All 270+ model types |
| **Failure mode** | Missing ops, shape errors, broken component wiring |
| **File** | `tests/build_graph_test.py` with configs from `tests/_test_configs.py` |
| **Marker** | None (always runs) |

L1 is the workhorse.  Every model type has at least one config entry.
Models with unique behaviour (MoE routing, sliding attention, partial
rotary) are flagged `is_representative=True` and always run; others are
skippable via `--fast`.  New registrations without an explicit config
entry get an auto-generated `(model_type, {}, False)` entry for
text-generation models.

#### What L1 checks

- Graph has expected inputs (`input_ids`, `attention_mask`,
  `position_ids`, `past_key_values.*`)
- Graph has expected outputs (`logits`, `present.*`)
- Initializers exist (embedding, attention projections, MLP weights)
- Node types are correct (e.g., `MatMul` count, `Attention` op present)

#### Supplementary L1 tests

| File | Purpose |
|------|---------|
| `tests/weight_alignment_test.py` | `preprocess_weights()` maps all HF keys to ONNX initializers |
| `tests/onnx_checker_test.py` | ONNX `CheckerPass` validates all built models |
| `tests/cli_test.py` | CLI entry points work with `--no-weights` |
| `src/**/*_test.py` | Co-located component unit tests (1000+ methods) |

#### Known limitation: tiny config false positives

A model might work with the real config but fail with the tiny config
(e.g., `num_heads=1` causes different GQA behavior, or `hidden_size=64`
is too small for certain attention patterns).  When this happens, either
adjust the tiny config to avoid the edge case, or mark the test with
`xfail` and a note explaining the divergence.  L2 (real config) catches
this gap.

### L2 — Architecture Validation

| Aspect | Detail |
|--------|--------|
| **Question** | Does a real HuggingFace `config.json` produce a valid full-size graph? |
| **What runs** | Download `config.json` (not weights) from HF Hub; build full-size graph; validate structure |
| **Speed** | ~5s per model (network for config only) |
| **Coverage** | All registered models with a `test_model_id` |
| **Failure mode** | Config field mismatch, missing field extraction, shape incompatibility at full scale |
| **File** | `tests/arch_validation_test.py` |
| **Marker** | `@pytest.mark.arch_validation` |

L2 catches bugs that L1 misses because L1 uses synthetic tiny configs.
A model might build at hidden=64 but fail at hidden=4096 because of
hardcoded assumptions.  L2 uses the real HF config shape without
downloading multi-GB weight files.

Note: L2 is **not** about ONNX graph validity (L1 already covers that).
L2 is about **config compatibility at real scale**.  Examples of
L2-only failures:

- A config field name changed in a newer HF model version
- A shape computation works at hidden=64 but hits alignment issues at
  hidden=7168
- Missing handling of optional config fields that real models set but
  tiny configs omit

### L3 — Synthetic Parity

| Aspect | Detail |
|--------|--------|
| **Question** | Does the ONNX graph compute the same function as HuggingFace PyTorch? |
| **What runs** | Build tiny ONNX + tiny HF model with identical random weights; compare single-forward-pass logits |
| **Speed** | 3–10s per model |
| **Coverage** | All causal LM model types (~200+) |
| **Failure mode** | Op-level bugs, wrong attention scaling, norm type mismatch, weight mapping errors |
| **File** | `tests/synthetic_parity_test.py` |
| **Marker** | None (runs in default suite) |

L3 is the **highest-ROI test tier** — it proves functional equivalence
for every model type without downloading weights or maintaining golden
files.

#### How it works

```
1. Build tiny ArchitectureConfig (hidden=64, 2 layers, 256 vocab)
2. Create HuggingFace model from AutoConfig with same dimensions
3. Initialize HF model with deterministic seed (torch.manual_seed(42))
4. Transfer HF state_dict → ONNX via preprocess_weights() + apply_weights()
5. Run forward(input_ids=[random 3 tokens]) on both
6. Compare logits with atol=1e-3, rtol=1e-3
```

#### What L3 catches that L1 cannot

- Attention scale `1/sqrt(head_dim)` vs custom `attention_multiplier`
- RMSNorm vs LayerNorm confusion (max abs diff > 1.0)
- Gate-up MLP wiring (swapped gate/up projections)
- Weight mapping bugs in `preprocess_weights()` (names shuffled)
- RoPE frequency computation errors

#### Known limitations

- Random weights don't exercise real value ranges (no overflow/underflow)
- Random weights can produce different precision mismatches than real
  weights (e.g., values near zero trigger denormalized float arithmetic).
  L4 (real weights) is the tier that catches precision-specific issues.
- Tiny models skip deep-layer interactions (accumulated RoPE drift)
- Models requiring `trust_remote_code=True` must be skipped
- MoE routing may differ between ONNX TopKGate and HF's router

Models with known divergences are marked `xfail(strict=False)` so they
still run — an XPASS signals the model is fixed and should be removed
from the skip list.

### L4 — Checkpoint Verified

| Aspect | Detail |
|--------|--------|
| **Question** | Does a specific HF checkpoint produce matching logits through the full pipeline? |
| **What runs** | Build ONNX model with real weights from HF Hub; run single prefill forward pass; compare argmax against pre-computed golden reference |
| **Speed** | Minutes (weight download + inference) |
| **Coverage** | ~50 representative checkpoints |
| **Failure mode** | Weight loading corruption, dtype casting errors, real-value numerical instability |
| **File** | `tests/e2e_golden_test.py` (class `TestL4CheckpointVerified`) |
| **Marker** | `@pytest.mark.golden` + `@pytest.mark.integration` |

L4 is **data-driven**: each test case is a YAML file in
`testdata/cases/`, and golden reference data is stored as `.npz` files
in `testdata/golden/`.  Adding test coverage = adding a YAML + `.npz`
file.  No code changes needed.

#### Gate criterion

**Argmax (top-1 token) must match.**  If the ONNX model's highest-logit
token at the last position differs from the golden reference, the test
fails.  This is the user-visible failure mode: the model would generate
different text.

Near-tied logits (gap between top-1 and top-2 below a threshold) are
flagged as AMBIGUOUS rather than FAIL to avoid false positives.

#### Golden file contents

```
top1_id, top2_id, top10_ids        ← gate data
top10_logits, logits_summary       ← diagnostic data
input_ids                          ← reproducibility
```

Full-vocab logits are NOT stored (~128 KB per model).  They can be
regenerated on demand via `scripts/generate_golden.py`.

### L5 — Generation E2E

| Aspect | Detail |
|--------|--------|
| **Question** | Does greedy autoregressive generation produce the correct token sequence? |
| **What runs** | Build ONNX model; run multi-step greedy decode loop; compare generated token sequence against golden reference |
| **Speed** | Minutes (weight download + multi-step inference) |
| **Coverage** | ~20 representative checkpoints |
| **Failure mode** | KV cache bugs, position ID drift, EOS handling, accumulated numerical errors |
| **File** | `tests/e2e_golden_test.py` (class `TestL5GenerationE2E`) |
| **Marker** | `@pytest.mark.generation` + `@pytest.mark.integration` |

L5 extends L4 by running the full autoregressive loop.  This catches
bugs that only manifest across multiple decode steps, such as:

- KV cache not properly carried between steps
- Position IDs not incrementing correctly
- Attention mask not growing with generated tokens
- Accumulated floating-point drift causing token divergence

#### Gate criterion

- **float32**: All generated tokens must match exactly
- **float16**: ≥90% token match ratio (allows ≤2 mismatches per 20 tokens)
- **bfloat16**: ≥80% token match ratio
- **int4**: ≥70% token match ratio

---

## 2. Test Pyramid and Coverage Strategy

### Tier coverage matrix

| Tier | Causal LM | Encoder | Seq2Seq | Vision | Audio | VL/Multimodal | MoE | Hybrid SSM |
|------|-----------|---------|---------|--------|-------|---------------|-----|------------|
| L0 | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All |
| L1 | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All |
| L2 | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All | ✅ All |
| L3 | ✅ All | ❌ Planned | ❌ Planned | ❌ N/A | ❌ Planned | ❌ N/A | ✅ All | ✅ All |
| L4 | ~30 | ~5 | ~5 | ~2 | ~3 | ~5 | ~3 | ~2 |
| L5 | ~15 | N/A | ~3 | N/A | ~2 | ~3 | ~1 | ~1 |

L3 currently covers causal LM models (the largest category).  Extending
to encoder, seq2seq, and audio models is feasible with the same pattern
and is planned for future work.

### Variant coverage strategy

Models within a family can exercise different code paths.  The strategy
is: **all variants at the cheapest tier, one representative at the
expensive tiers.**

```
L1 (graph build, <1s):       ALL variants × ALL tasks    ← cheap, go broad
L3 (synthetic parity, ~5s):  ALL variants × f32          ← medium cost, still broad
L4 (golden, minutes):        ONE representative × f32    ← expensive, be selective
L5 (generation, minutes):    ONE representative × f32    ← most expensive, minimal
```

Known model families requiring multiple L1/L3 configs:

| Family | Variant Triggers | Config Count |
|--------|-----------------|--------------|
| Qwen3/3.5 | `layer_types` (dense vs linear-attention), MoE | 3 |
| DeepSeek v2/v3 | `use_mla`, MoE | 2 |
| Falcon-H1 | Mamba vs attention blocks | 2 |
| Jamba/Bamba | SSM + attention hybrid layers | 2 each |
| Gemma3n | `per_layer_config_type` (sliding vs global) | 2 |
| Granite | `attention_multiplier`, `logits_scaling` | 2 |

---

## 3. Parity Metrics

### Severity-tiered evaluation

Rather than a single `atol`/`rtol` check, comparisons use a
**three-tier evaluation** where each tier maps to a user need:

#### Tier 1 — Functional correctness ("does it work?")

- **Top-1 (argmax) token agreement** at every decode position
- Pass/fail gate: if argmax mismatches, the model generates different
  text — this is a hard failure
- Near-tied logits (gap < threshold) produce AMBIGUOUS, not FAIL

#### Tier 2 — Distribution correctness ("is it trustworthy?")

- **Top-K agreement** (K=10): Jaccard similarity of top-K token sets
- **Cosine similarity** of full logit vectors
- These catch "accidentally correct" cases where argmax agrees but the
  distribution is shifted, affecting sampling-based generation

#### Tier 3 — Numerical precision ("is it exact?")

- `max_abs_diff`, `mean_abs_diff`, traditional `atol`/`rtol`
- Summary statistics: `logits_mean`, `logits_std`, `logits_l2_norm`
- For debugging, regression detection, and ensuring no systematic bias

#### Evaluation order

```
1. Does argmax match?    → CRITICAL — if no, stop and report failure
2. Does top-K overlap?   → WARNING  — if low, flag for investigation
3. Within atol?          → INFO     — for precision tracking
```

### Per-dtype thresholds

| Dtype | Argmax | Top-10 Jaccard | Cosine Sim | atol |
|-------|--------|----------------|------------|------|
| float32 | must match | ≥ 0.9 | ≥ 0.9999 | 1e-4 |
| float16 | must match | ≥ 0.8 | ≥ 0.999 | 5e-3 |
| bfloat16 | must match | ≥ 0.7 | ≥ 0.99 | 1e-2 |
| int4 | ≤ 2 mismatches / 20 tokens | ≥ 0.5 | ≥ 0.95 | 0.5 |

### Metrics per test tier

| Tier | Primary Metric | Rationale |
|------|---------------|-----------|
| L3 (Synthetic) | argmax + atol (1e-3) | Random weights → unpredictable scale → tolerance-based |
| L4 (Golden) | argmax match + top-K Jaccard | Stored reference → rank-based is stable |
| L5 (Generation) | exact token match (f32) or match ratio | Functional correctness is all-or-nothing |

---

## 4. CI Integration

### Execution cadence

| Trigger | What runs | Time budget | Purpose |
|---------|-----------|-------------|---------|
| **Every PR** | L1 (smoke) + L3 (synthetic parity, selective) | < 5 min | Fast developer feedback |
| **Push to main** | L1 + L3 + selective L4 (fast golden) | < 10 min | Post-merge verification |
| **Nightly** | L2 (architecture validation) | ~30 min | Config compatibility |
| **Weekly (GPU)** | L4 (all golden) + L5 (generation) | ~60 min | Full checkpoint verification |
| **Weekly (Sun)** | Golden file regeneration | ~90 min | Keep golden data current |

### CI workflow mapping

| Workflow File | Tiers | Schedule |
|---------------|-------|----------|
| `main.yml` | L1, L3, selective L4/L5 | PR + push to main |
| `gpu_tests.yml` | L4, L5 | Push to main + weekly |
| `nightly_l2.yml` | L2 | Nightly (3 AM UTC) |
| `golden_regen.yml` | — | Weekly (Sun 2 AM UTC) + manual |
| `benchmark.yml` | Performance | PR + manual |

### Feedback timing targets

| Event | Time to know | How |
|-------|-------------|-----|
| Graph broke | < 2 min | L1 in PR CI |
| Math is wrong | < 5 min | L3 synthetic parity in PR CI |
| Golden file regressed | < 5 min | L4 golden comparison in PR CI (no downloads) |
| Real model broken | Nightly / weekly | L4/L5 on schedule |

### Pytest markers

| Marker | Tier | Usage |
|--------|------|-------|
| *(none)* | L1, L3 | Default — always runs |
| `arch_validation` | L2 | `pytest -m arch_validation` |
| `golden` | L4 | `pytest -m golden` |
| `generation` | L5 | `pytest -m generation` |
| `integration` | L4, L5 | Superset — `pytest -m integration` |
| `integration_fast` | Subset of L4 | Small models only |
| `integration_slow` | Subset of L4 | Large/multimodal models |
| `benchmark` | Perf | `pytest -m benchmark` |

### CLI flags

| Flag | Effect |
|------|--------|
| `--fast` | Skip non-representative configs (L1 runs ~20 models in ~5s) |
| `--models qwen2,llama` | Run only specified model types |
| `-m "not integration"` | Exclude all weight-downloading tests |

---

## 5. Test Data Infrastructure

### Directory layout

```
testdata/
├── cases/                    # YAML test case definitions (L4/L5)
│   ├── causal-lm/
│   ├── vision-language/
│   ├── encoder/
│   ├── seq2seq/
│   ├── audio/
│   └── vision/
├── golden/                   # Pre-computed HF reference outputs (.npz)
│   ├── causal-lm/
│   ├── vision-language/
│   └── ...
├── default_tolerances.yaml   # L3/L4/L5 comparison thresholds
└── ...                       # Audio/image fixtures
```

### Data-driven test design

The golden test infrastructure follows a single principle: **adding
test coverage = adding a YAML file**.

1. Create a YAML file in `testdata/cases/<task-type>/` describing the
   model, inputs, and expected level
2. Run `python scripts/generate_golden.py --case <path>` to produce
   the `.npz` reference
3. Commit both files
4. `tests/e2e_golden_test.py` auto-discovers and parametrizes

Missing golden files produce pytest SKIP, not FAIL — enabling a
YAML-first workflow where intent (YAML) can be committed before data
(`.npz`).

### Tolerance configuration

Tolerances are centralized in `testdata/default_tolerances.yaml` with
per-level, per-dtype thresholds.  Individual YAML test cases can
override specific thresholds when a model has known precision
characteristics.

---

## 6. Developer Workflow

### Adding a new model

```
1. Create model class → L1 passes automatically
2. Add config entry in _test_configs.py
3. Run:  pytest tests/build_graph_test.py -k "my_model"        → L1
4. Run:  pytest tests/synthetic_parity_test.py -k "my_model"   → L3
5. Register test_model_id → L2 passes on nightly
6. Add YAML + generate golden → L4/L5 follow
```

L3 synthetic parity is the **primary development-loop test** — the
test contributors run repeatedly while debugging.  It requires no
network access, no golden files, no weight downloads, and completes
in < 10 seconds.

### Debugging numerical mismatches

When a test fails, the diagnostic output should be actionable:

```
✅ Top-1 token matches (token 12366 "Paris")
✅ 9/10 top-10 tokens match
⚠️  Logit atol=1.2e-3 exceeds threshold 1e-3
    → Functional behavior preserved; precision difference likely acceptable for f16
```

Common root causes by diagnostic pattern:

| Pattern | Likely cause |
|---------|-------------|
| max abs diff > 10 | Weights loaded to wrong parameters |
| max abs diff > 0.5 | Norm type mismatch (RMSNorm vs LayerNorm) |
| max abs diff > 0.1, correct argmax | Attention scale difference |
| Argmax mismatch, all logits shifted | Missing embedding/logit scaling multiplier |
| Argmax only wrong at long positions | RoPE frequency computation error |

### Debugging multi-model pipelines (VLM, TTS)

For multi-model tasks, isolate each model boundary:

1. Compare each model's output against HF at the boundary (e.g.,
   vision encoder `last_hidden_state`, projector output)
2. Check pre-norm vs post-norm — HF's `outputs.last_hidden_state` is
   typically post-norm
3. Verify external input construction matches HF (e.g., concatenating
   hidden states with embeddings for code predictors)

---

## 7. Performance Regression Testing

Performance testing complements correctness testing with four build
metrics tracked per model:

| Metric | Warning | Blocker | Enforcement |
|--------|---------|---------|-------------|
| Build time | >10% | >25% | Advisory (timing noise) |
| Peak memory | >10% | >25% | Advisory |
| ONNX node count | >5% | >10% | **Can gate merge** (deterministic) |
| Model size (bytes) | >5% | >10% | **Can gate merge** (deterministic) |

Only deterministic metrics (node count, model size) can block PRs.
Timing metrics are shown in PR comments but never cause merge failures
due to CI runner noise.

See [Performance Benchmarking](perf-benchmarking.md) for full details.

---

## 8. Current State and Gaps

### What exists today

| Tier | Status | File |
|------|--------|------|
| L0 | ✅ Implicit via registry | `_registry.py` |
| L1 | ✅ Full coverage (270+) | `build_graph_test.py` |
| L2 | ✅ Nightly CI | `arch_validation_test.py` |
| L3 | ✅ Causal LM (~200 types) | `synthetic_parity_test.py` |
| L4 | ✅ ~50 checkpoints | `e2e_golden_test.py` |
| L5 | ✅ ~20 checkpoints | `e2e_golden_test.py` |
| Perf | ⚠️ Script exists, CI not wired | `benchmark_build.py` |

### Known gaps

| Gap | Impact | Plan |
|-----|--------|------|
| L3 covers only causal LM | Encoder/seq2seq/audio models lack parity tests | Extend `synthetic_parity_test.py` to other task types |
| ~40 L3 models are xfail | MoE routing, GPT-2 family, Gemma softcapping diverge | Fix per-model; track in xfail list |
| L4/L5 golden files need periodic refresh | HF Transformers updates may change reference outputs | Weekly `golden_regen.yml` workflow |
| Performance CI not wired | No regression detection for node count / build time | Wire `benchmark.yml` (see perf-benchmarking.md) |
| No cross-platform golden tolerance | CPU vs GPU golden files may differ | Store platform tolerance bands |

---

## 9. Quick Reference Commands

```bash
# L1: Smoke (all models, ~30s)
pytest tests/build_graph_test.py -v

# L1: Fast (representative only, ~5s)
pytest tests/build_graph_test.py --fast

# L1: Single model
pytest tests/build_graph_test.py -k "qwen2"

# L2: Architecture validation (requires network)
pytest tests/arch_validation_test.py -m arch_validation -v

# L3: Synthetic parity (all causal LM, ~10 min)
pytest tests/synthetic_parity_test.py -v --tb=short -n 0

# L3: Single model parity
pytest tests/synthetic_parity_test.py -k "llama" -v

# L4: Golden logits (requires weights)
pytest tests/e2e_golden_test.py -m golden -v

# L5: Generation E2E (requires weights)
pytest tests/e2e_golden_test.py -m generation -v

# All non-integration tests
pytest -m "not integration" --tb=short

# Filter by model type
pytest tests/build_graph_test.py --models "qwen2,llama"

# Regenerate golden files
python scripts/generate_golden.py --force
```

---

## 10. Distributed GPU Testing & Sharding

### Problem

L4 (golden logits) and L5 (generation E2E) tests require real model
weights and GPU inference.  Today, `gpu_tests.yml` runs on a single
`gpu` runner sequentially.  As golden test cases grow (one YAML per
model family), a single machine becomes a bottleneck — each model
download + inference can take 2–5 minutes, and the workflow already
has a 60-minute timeout.

### Sharding strategy

The recommended approach is **task-type directory sharding**: split
the `testdata/cases/` YAML files into per-category directories
(`text-generation/`, `vision-language/`, `speech/`, etc.) and assign
each directory to a separate GPU runner in a matrix job.

```yaml
# gpu_tests.yml — proposed matrix sharding
jobs:
  golden-tests:
    strategy:
      fail-fast: false
      matrix:
        shard:
          - { name: "text-gen",    filter: "testdata/cases/text-generation/" }
          - { name: "vlm",         filter: "testdata/cases/vision-language/" }
          - { name: "speech",      filter: "testdata/cases/speech/" }
          - { name: "other",       filter: "testdata/cases/other/" }
    runs-on: gpu
    steps:
      - name: Run L4/L5 for shard
        run: |
          pytest tests/e2e_golden_test.py \
            -m "golden or generation" \
            --cases-dir "${{ matrix.shard.filter }}" \
            -v --timeout=300
```

Benefits:
- **Linear scale-out**: Adding a GPU runner proportionally reduces
  wall-clock time.
- **Failure isolation**: A VLM failure doesn't block text-generation
  results.
- **HF cache efficiency**: Each shard downloads only the models it
  needs; `actions/cache` keys can be scoped per shard.

### pytest-xdist for within-shard parallelism

For shards with many small models (text-generation), pytest-xdist
with `--dist loadfile` groups tests from the same YAML file onto one
worker, preventing redundant model downloads:

```bash
pytest tests/e2e_golden_test.py -m golden -n 2 --dist loadfile \
  --cases-dir testdata/cases/text-generation/
```

### GPU runner pool design

| Pool     | Count | GPU Memory | Primary use          |
|----------|-------|-----------|----------------------|
| gpu-sm   | 2     | 16 GB     | Text-gen, encoder    |
| gpu-lg   | 1     | 48 GB+    | VLM, speech, MoE     |

The matrix shard-to-pool assignment is declared in the workflow so
that memory-intensive VLM models route to `gpu-lg` runners.

### Affected-model scoping on GPU

On PRs, the `detect-affected` job output can further narrow GPU tests.
The synthetic parity job already does this on CPU.  The same pattern
extends:

```yaml
- name: Run L4 for affected models only
  if: needs.detect-affected.outputs.run_all != 'true'
  run: |
    AFFECTED='${{ needs.detect-affected.outputs.affected }}'
    FILTER=$(echo "$AFFECTED" | python -c "
      import json, sys, re
      names = json.load(sys.stdin)
      print(' or '.join(re.sub(r'[^a-zA-Z0-9_]', '_', n) for n in names))
    ")
    pytest tests/e2e_golden_test.py -m golden -k "$FILTER" -v
```

---

## 11. End-to-End Testing with ONNX Runtime GenAI

### Motivation

Tiers L1–L5 verify the ONNX graph in isolation.  But the final
consumer is the onnxruntime-genai (ORT GenAI) inference runtime, which
requires a specific directory layout, a `genai_config.json`, and
optionally `processor_config.json`.  A model can pass all L1–L5 tiers
yet fail at runtime because of an incorrect config or missing tokenizer
file.

ORT GenAI testing validates the **last mile**: build → export →
configure → load → generate.

### Current infrastructure

**`auto_export()` pipeline** (`integrations/ort_genai/auto_export.py`):
Chains the full export sequence into a single call:

```python
auto_export(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir=Path("/tmp/qwen25vl"),
    # Optional: precision="fp16", device="cuda"
)
```

Steps performed:
1. `build(model_id)` → `ModelPackage` (dict of `str → ir.Model`)
2. Download & apply real weights from HuggingFace
3. Generate `genai_config.json` via `GenaiConfigGenerator` — includes
   model file mappings, search parameters, and for VLM models, the
   `vision` and `embedding` sections with image processing config
4. Save ONNX model files (one per component: decoder, vision, embedding)
5. Copy tokenizer files from the HuggingFace cache
6. Write `processor_config.json` for vision models (ort-extensions format)

**`GenaiConfigGenerator`** (`integrations/ort_genai/genai_config.py`):
Fluent builder that maps HuggingFace config fields to the ORT GenAI
JSON schema.  Handles:
- Model type mapping (`_ORT_GENAI_MODEL_TYPE`: maps HF types like
  `"qwen2_vl"` to ORT GenAI types like `"qwen2_vl"`)
- Multi-model sections: `model.vision`, `model.embedding` for VLM;
  `model.speech` for audio models
- Image size / processor configuration for vision pipelines

**`tests/ort_genai_test.py`**: Integration test pattern:

```python
class TestOrtGenaiQwen25VL:
    @pytest.mark.integration_slow
    def test_text_generation(self, tmp_path):
        # 1. Build 3-model package
        pkg = build("Qwen/Qwen2.5-VL-3B-Instruct")
        # 2. Apply weights
        apply_weights(pkg, model_id)
        # 3. Save flat layout
        for name, model in pkg.items():
            onnx.save(model, tmp_path / f"{name}.onnx")
        # 4. Write genai_config.json + processor_config.json
        _write_genai_config(tmp_path, model_params)
        _copy_tokenizer(tmp_path, model_id)
        # 5. Load into ORT GenAI runtime
        model = ort_genai.Model(str(tmp_path))
        # 6. Run greedy generation, assert output text
        output = generate(model, tokenizer, prompt)
        assert "expected" in output
```

### Config generation testing

A gap in the current strategy is **config-only validation**: verifying
that `GenaiConfigGenerator` produces valid JSON for all registered
model types without downloading weights or running inference.  This
can be a cheap L1-adjacent test:

```python
@pytest.mark.parametrize("model_type", registry.architectures())
def test_genai_config_generates(model_type, tmp_path):
    """Config generation should not crash for any registered type."""
    reg = registry.get_registration(model_type)
    config = reg.config_class(**tiny_overrides)
    generator = GenaiConfigGenerator(config)
    result = generator.generate()
    assert "model" in result
    # Validate required keys
    assert "decoder" in result["model"]
```

### VLM-specific validation

Vision-language models produce three ONNX files and require matching
`model.vision` and `model.embedding` sections in `genai_config.json`.
The test must verify:

1. All three model files exist and can be loaded by ORT
2. `genai_config.json` references correct filenames
3. Image preprocessing config (resize, normalize, crop) matches HF
4. Tokenizer and processor files are present and well-formed

### Proposed test tiers for ORT GenAI

| Level | What it tests                        | Cost     | CI cadence    |
|-------|--------------------------------------|----------|---------------|
| L1+   | Config generation (no weights)       | Seconds  | Every PR      |
| E2E   | Full pipeline: export + load + gen   | Minutes  | Weekly / GPU   |

---

## 12. New Model Onboarding & Coverage Enforcement

### The onboarding path

When a contributor adds a new model architecture, the test tiers
should automatically detect and require coverage.  The current system
provides progressively automated coverage:

**Automatic coverage (no manual test work needed):**
- **L0 (Registered)**: Achieved by adding the model to `_registry.py`.
  The dashboard auto-detects all registered model types via
  `_scan_registry()`.
- **L1 (Graph builds)**: Achieved by adding a tiny config to
  `_MODEL_CONFIGS` in `tests/_test_configs.py`.  The parametrized
  `build_graph_test.py` picks it up automatically.
- **L2 (Architecture validation)**: Achieved by setting `test_model_id`
  in the registry entry.  The `arch_validation_test.py` discovers it
  automatically.
- **L3 (Synthetic parity)**: All entries in `ALL_CAUSAL_LM_CONFIGS`
  from `_test_configs.py` are automatically tested in
  `synthetic_parity_test.py`.  No additional test code needed for
  standard text-generation models.

**Manual test work required:**
- **L4 (Golden logits)**: Add a YAML test case file under
  `testdata/cases/<category>/` specifying the HuggingFace model ID,
  input prompt, and expected logit snapshot.
- **L5 (Generation E2E)**: Add a generation test case to the same
  YAML file with expected output text.
- **Integration tests**: Required for non-standard architectures
  (VLM, speech, MoE) that need specialized validation.

### Checklist for new models

The `adding-a-new-model` skill documents the complete procedure:

1. Create `models/<name>.py` with model class
2. Export from `models/__init__.py`
3. Register in `_registry.py` with `test_model_id`
4. Add tiny config to `_MODEL_CONFIGS` in `tests/_test_configs.py`
5. Run `pytest tests/build_graph_test.py -k "<name>"` → L1
6. Run `pytest tests/synthetic_parity_test.py -k "<name>"` → L3
7. (Optional) Add YAML golden test case → L4/L5

### Ensuring coverage — automated enforcement

**Dashboard gap detection**: `generate_dashboard.py` computes
`confidence_level` per model.  Models stuck at L0 (registered but
never tested) are surfaced prominently.  The dashboard summary shows
`by_level` counts — a growing L0 count signals missing coverage.

**PR-scoped enforcement via `detect_affected_models.py`**:

When a PR modifies model source files, the `detect-affected` CI job
runs AST-based static analysis to identify affected model types:

1. `classify_file()` categorizes each changed file as `model`,
   `component`, `task`, `shared_infra`, `test`, or `other`
2. For `model` files, the script resolves the file → module name →
   registered model types via `_build_source_module_to_types()`
3. For transitive dependencies, `_find_reverse_dependents()` does BFS
   on the reverse import graph to find all modules that depend on the
   changed file
4. If any `shared_infra` file changes (components/, tasks/,
   `_registry.py`, `_configs.py`, etc.), the output is `run_all: true`
   and all tests execute

The affected model list is passed to downstream CI jobs (synthetic
parity, golden comparison) as a pytest `-k` filter, so only relevant
models are tested on PRs — keeping CI fast while ensuring coverage.

**Missing coverage detection**: A future CI check could fail PRs that
add a new registry entry without a corresponding `_test_configs.py`
entry.  This ensures L1 coverage before merge:

```yaml
- name: Check new models have test configs
  run: |
    python -c "
    from mobius._registry import registry
    from _test_configs import ALL_CONFIGS
    tested = {mt for mt, _, _ in ALL_CONFIGS}
    untested = set(registry.architectures()) - tested
    # Allow known exceptions
    untested -= KNOWN_UNTESTED
    if untested:
        print(f'Models missing test configs: {untested}')
        exit(1)
    "
```

---

## 13. Confidence Dashboard Design

### Purpose

The confidence dashboard provides a single-page view of testing
coverage across all 270+ model types.  It answers the question:
**"How confident are we that model X works correctly?"** — not just
whether the graph builds, but whether it produces correct numerics
and generates valid text.

### Architecture

**Generator** (`scripts/generate_dashboard.py`):
A Python script that produces a self-contained HTML file with no
external dependencies (all CSS/JS inline).  It:

1. Scans the model registry for all registered architectures
2. Scans each test tier (L1–L5) using tier-specific `_scan_*` functions
3. Collects metadata: code paths exercised, config overrides, YAML
   test cases, xfail/skip reasons
4. Computes summary statistics and family groupings
5. Embeds everything as JSON in the HTML for client-side filtering

**Data model** — `ModelInfo` dataclass per model type:

| Field               | Source                        | Description                     |
|---------------------|-------------------------------|---------------------------------|
| `l1_graph_build`    | `_test_configs.ALL_CONFIGS`   | Has a tiny-config build test    |
| `l2_arch_validation`| `test_model_id` in registry   | HF config compatible            |
| `l3_synthetic_parity`| `ALL_CAUSAL_LM_CONFIGS`      | Random-weight parity tested     |
| `l3_status`         | `_SKIP_REASONS`/`_XFAIL_REASONS` | pass / xfail / skip         |
| `l4_golden_files`   | `testdata/golden/*.json`      | Logit snapshot exists           |
| `l5_generation_golden`| `testdata/golden/*_generation.json` | Text output snapshot   |
| `l4_has_test_case`  | `testdata/cases/**/*.yaml`    | YAML-driven L4 test case        |
| `l5_has_test_case`  | `testdata/cases/**/*.yaml`    | YAML-driven L5 test case        |
| `code_paths`        | `detect_code_paths(overrides)`| Variant features: MoE, GQA, etc. |
| `has_integration_test`| `*integration*.py` scan     | Mentioned in integration tests  |
| `confidence_level`  | Computed: max tier achieved   | 0–5 integer                     |

### Information surfaced

**Hero metrics** (top of dashboard):
- Total registered model types (e.g. "274 models")
- Distribution by confidence level (bar chart): how many at L0, L1,
  L2, L3, L4, L5
- L3 parity status breakdown: pass / xfail / skip / untested

**Per-model drill-down** (searchable, filterable table):
- Model type, module class, task, family
- Tier indicators (L1–L5 checkmarks)
- L3 status with reason (xfail tooltip shows the bug description)
- Code paths exercised (e.g. "moe", "gqa", "sliding_window")
- YAML test case file link
- Config overrides used in build tests

**Category breakdown**: Groups models by task type (Causal LM, Vision,
Vision-Language, Speech, Encoder, Seq2Seq, etc.) with per-category
confidence distributions.

**Family grouping**: Models with shared prefixes (qwen2, qwen2_moe,
qwen2_vl → family "qwen2") are grouped so that family-level minimum
confidence is visible.  The `_derive_family()` function uses a
prefix table (`_FAMILY_PREFIXES`) ordered longest-first.

**Code-path coverage**: Shows which variant features (MoE routing,
GQA, sliding window, RoPE variants) are exercised by test configs,
helping identify untested code branches.

### CI / PR integration

**Push-to-main regeneration**: The `pages.yml` workflow regenerates
the dashboard on every push to `main` and deploys to GitHub Pages.
This ensures the dashboard always reflects the current state of the
codebase.

```yaml
# pages.yml
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6am UTC
```

**PR impact preview** (proposed): A lightweight CI step could generate
a dashboard diff on PRs, showing how the PR changes model coverage:

```yaml
- name: Dashboard diff
  run: |
    # Generate dashboard for base branch
    git stash
    python scripts/generate_dashboard.py --json base.json
    git stash pop
    # Generate dashboard for PR branch
    python scripts/generate_dashboard.py --json pr.json
    # Diff: new models, level changes, new xfails
    python scripts/dashboard_diff.py base.json pr.json >> $GITHUB_STEP_SUMMARY
```

**Detecting modeling code updates**: The `detect_affected_models.py`
script identifies which model types are impacted by source changes.
Integrating its output into the dashboard diff would show reviewers
exactly which models need attention:

- "This PR modifies `models/qwen2.py` → affects 5 model types:
  qwen2, qwen2_moe, qwen2_audio, qwen2_vl, qwen2_5"
- "Affected models test status: 3 at L3, 1 at L4, 1 at L1"

### Dashboard evolution roadmap

| Phase | Feature                          | Status      |
|-------|----------------------------------|-------------|
| 1     | Static HTML with L1–L5 tiers     | Implemented |
| 2     | L3 xfail/skip status tracking    | Implemented |
| 3     | YAML test case scanning (L4/L5)  | Implemented |
| 4     | Code-path variant coverage       | Implemented |
| 5     | Family grouping                  | Implemented |
| 6     | PR dashboard diff                | Proposed    |
| 7     | CI artifact linking (JUnit XML)  | Proposed    |
| 8     | Historical trend tracking        | Future      |

---

## Open Questions

1. **L3 breadth**: Should synthetic parity extend to encoder-only and
   seq2seq models?  The infrastructure exists; the main effort is
   writing task-specific comparison logic.

2. **Seed sensitivity**: Should L3 run with multiple random seeds
   (proposal: 3) to detect near-tie sensitivity?

3. **xfail reduction**: The ~40 xfailed L3 models represent real
   implementation gaps (MoE routing, GPT-2 family norms, Gemma
   softcapping).  Should these be prioritized as bug fixes or accepted
   as known limitations?

4. **Code-path tags**: Should test configs carry explicit code-path
   tags (e.g., `{"code_paths": ["linear_attn", "moe"]}`) to make
   variant coverage measurable on the dashboard?

5. **Auto-discovery**: Should a static analysis tool auto-detect
   variant-triggering config fields from `if config.X` branches in
   model code?
