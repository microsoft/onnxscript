# Testing Strategy Analysis

**Date:** 2026-03-05
**Scope:** Multi-layer testing strategy for mobius (273 registered model types)

> This document synthesizes two independent analyses of the testing strategy
> design questions: one from a **product-manager perspective** (practical,
> user-persona-driven, prioritized for impact) and one from a
> **radical-thinker perspective** (first-principles, assumption-challenging,
> creative). Where the two perspectives converge, recommendations are
> strong. Where they diverge, both viewpoints are presented.

---

## 1. Testing Tiers: Synthetic Parity as a First-Class Layer

### The core question

Should tiny random-weight parity (ONNX vs HuggingFace on synthetic inputs)
be a sub-level of the existing L3 golden-file parity, or a distinct testing
tier?

### Consensus: they test different things

Both analyses agree that synthetic parity and golden-file parity answer
**fundamentally different questions** and should not be nested:

| Test | What it proves | What it cannot prove |
|---|---|---|
| **Synthetic parity** (random weights, live ONNX vs HF) | The ONNX graph computes the **same function** as HF PyTorch for this architecture | Weight loading works; real-value-range numerical stability |
| **Golden-file parity** (real weights, ONNX vs stored reference) | A specific HF checkpoint converts correctly end-to-end | That HF reference is still current; regressions in HF itself |

Synthetic parity is a **developer test** ("is my math right?"). Golden-file
parity is a **user trust test** ("does this checkpoint work in production?").

### Recommended tier structure

| Level | Name | What it tests | Speed | Coverage |
|---|---|---|---|---|
| L0 | Registered | Model type exists in registry | instant | 273/273 |
| L1 | Graph Construction | ONNX graph builds, ops are valid | <1s | 273/273 |
| L2 | Architecture Validation | Real HF config shapes are consistent | ~5s | 273/273 |
| **L3** | **Synthetic Parity** | **ONNX = HF on tiny random-weight models** | **3–10s** | **273/273** |
| L3+ | Checkpoint Verified | ONNX matches stored reference with real weights | minutes | ~50/273 |
| L4 | Generation E2E | Full autoregressive loop produces correct text | minutes | ~50/273 |

Key design choices in this numbering:

- **L3 is the broad parity layer.** It covers all 273 model types with no
  network access, no golden files, and no staleness problem. It is the
  highest-ROI addition to the current plan.
- **L3+ uses the "+" notation** to communicate that it includes everything L3
  proves, *plus* verification against a real checkpoint. It is additive, not
  a separate dimension.
- **L4 remains the end-to-end generation test.** Exact token-sequence match
  for greedy decoding.

### Why L3 (synthetic parity) at full breadth is feasible

The existing `integration_test.py` already demonstrates the pattern: create
an HF model from config (tiny), build an ONNX model from the same config,
transfer weights via `preprocess_weights()` + `apply_weights()`, and compare
outputs. This takes ~2–5 seconds per model. For 273 models, that is
~10–20 minutes total — comparable to the existing L2 time budget.

The infrastructure already exists: `_test_configs.py` has tiny configs for
every model type, and the `TestGeneration` class builds models with random
weights in under one second.

### Tradeoffs to acknowledge

- **Random weights don't exercise real value ranges.** Weights initialized at
  small scale (e.g. 0.02) won't trigger overflow/underflow that real weights
  might. Mitigation: use Xavier-initialized weights for realistic scale.
- **Tiny models skip deep-layer interactions.** A 2-layer model can't catch
  bugs that only manifest with many layers (e.g. accumulated RoPE drift).
  This is what L3+ golden tests are for.
- **Weight transfer is the critical path.** Synthetic parity relies on
  `preprocess_weights()` correctly mapping HF weight names to ONNX parameter
  names. If this mapping has bugs, the test fails even though the graph is
  correct. This is actually a *feature* — weight mapping bugs are real bugs
  — but error messages should distinguish numerical divergence from weight
  mapping failures.
- **Seed sensitivity.** A test that passes with one random seed might fail
  with another if random weights create near-tied logits. Mitigation: run
  with 3 different seeds; if only 1/3 fails, flag as a near-tie sensitivity
  issue rather than a hard failure.

---

## 2. Parity Metrics: What "Correct" Means

### The problem with `assert_allclose`

Both analyses converge on the same critique: a single `atol`/`rtol` check is
simultaneously **too strict** and **too loose**.

- **Too strict for quantized models.** What does `atol=0.5` mean when
  comparing a 4-bit approximation to a 32-bit reference?
- **Too loose for functional correctness.** Two logit vectors with a tiny
  difference can flip the argmax (and thus the generated text), while vectors
  shifted by a large constant produce identical softmax distributions.
- **A single pass/fail bit.** When allclose fails, you get a wall of numbers
  with no indication of whether the model is fundamentally broken or slightly
  imprecise.

### Recommended metric hierarchy

Replace the single tolerance check with a **severity-tiered evaluation**
where each tier maps to a user need:

#### Tier 1 — Functional correctness ("does it work?")

- **Top-1 (argmax) token agreement** at every position.
- Pass/fail gate: argmax must match for prefill's last position and all
  decode tokens.
- If this fails, the model generates different text. This is the
  user-visible failure mode.

#### Tier 2 — Distribution correctness ("is it trustworthy?")

- **Top-K agreement** (K=10): Jaccard similarity of top-K token sets.
- **Cosine similarity** of full logit vectors.
- **KL divergence**: `KL(softmax(onnx) || softmax(hf))` — the
  information-theoretically correct metric for comparing probability
  distributions, and meaningful across all dtypes.
- This tier catches "accidentally correct" cases where argmax agrees but the
  distribution is shifted, which affects sampling-based generation
  (temperature, top-p).

#### Tier 3 — Numerical precision ("is it exact?")

- `max_abs_diff`, `mean_abs_diff`, and traditional `atol`/`rtol`.
- Summary statistics: `logits_mean`, `logits_std`, `logits_l2_norm`.
- For debugging, regression detection, and ensuring ONNX ops don't introduce
  systematic bias.

### Evaluation order matters

```
1. Does argmax match?     → CRITICAL — if no, stop and report failure loudly
2. Does top-K overlap?    → WARNING  — if low, flag for investigation
3. Are logit values within tolerance? → INFO — for precision tracking
```

Failure messages should be actionable, not walls of numbers:

```
✅ Top-1 token matches (token 12366 "Paris")
✅ 9/10 top-10 tokens match
⚠️  Logit atol=1.2e-3 exceeds threshold 1e-3 (but functional behavior preserved)
    → Consider: this may be an acceptable precision difference for f16
```

### Per-dtype thresholds

| Dtype | Argmax | Top-10 Jaccard | Cosine Sim | KL Divergence | atol |
|---|---|---|---|---|---|
| f32 | must match | ≥ 0.9 | ≥ 0.9999 | ≤ 1e-4 | 1e-4 |
| f16 | must match | ≥ 0.8 | ≥ 0.999 | ≤ 1e-2 | 5e-3 |
| bf16 | must match | ≥ 0.7 | ≥ 0.99 | ≤ 5e-2 | 1e-2 |
| int4 (GPTQ/AWQ) | ≤ 2 mismatches / 20 tokens | ≥ 0.5 | ≥ 0.95 | ≤ 0.5 | 0.5 |

For quantized models, the definition of "correct" relaxes to "functionally
similar" rather than "numerically identical." This matches user expectations.

### Metrics per test level

| Test Level | Primary Metric | Rationale |
|---|---|---|
| L3 (Synthetic Parity) | argmax + top-K Jaccard | Random weights → logit scale is unpredictable → rank-based metrics are scale-invariant |
| L3+ (Checkpoint Verified) | Full multi-metric suite | Detailed comparison against a known reference — report everything |
| L4 (Generation E2E) | Exact token match (f32/f16) or Levenshtein distance (int4) | The ultimate functional test — tokens either match or they don't |

### Edge case: near-tied logits

When the gap between top-1 and top-2 logits is below a threshold (e.g. 0.01
for f32), argmax becomes non-deterministic. The model has equal confidence in
two tokens. When this happens:

- Flag the comparison as **"ambiguous"** rather than "fail."
- In generation tests, allow the first divergence point to be a "soft
  mismatch" if the logit gap was small.
- The dashboard should surface these separately from hard failures.

---

## 3. Architecture Variants and Code-Path Coverage

### The variant problem

Models within a family can exercise fundamentally different code paths.
Qwen3.5, for example, has dense, linear-attention (DeltaNet), and MoE
variants — three distinct computational graphs under one family name.

A single test config might cover paths A + B + C + D, but you won't know if
path B alone is correct unless you test a config that *only* uses B.

### Two complementary approaches

#### Approach A: Family-level grouping on the dashboard (user-facing)

Users think in terms of **model families**, not `model_type` strings.
The dashboard should group by family with drill-down:

```
Model Family     | Variants | Confidence | f32  | f16
─────────────────┼──────────┼────────────┼──────┼─────
Qwen3            │ 2        │ 🟢 L4     │ L4   │ L3
  └ qwen3 (dense)│          │ 🟢 L4     │ L4   │ L3
  └ qwen3_moe    │          │ 🟢 L3+    │ L3+  │ —
Llama            │ 1        │ 🟢 L4     │ L4   │ L3
```

Key UX decisions:

- **Family confidence = minimum across variants.** Conservative on purpose —
  prevents users from assuming MoE coverage when only dense is tested.
- **Derive family grouping automatically** from registry entries. Strip
  common suffixes (`_moe`, `_text`, `_vl`, `_text_only`), group by prefix,
  and allow an explicit `family` override in `ModelRegistration` for edge
  cases.
- **Flag incomplete families**: "⚠️ Qwen3: 1 of 2 variants untested."

#### Approach B: Code-path tagging in test configs (developer-facing)

Instead of enumerating model names, tag the **code paths** each config
exercises. This makes variant coverage visible and measurable:

```python
# In _test_configs.py
("qwen3_next", {layer_types: ["linear", "full"], num_local_experts: 4}, True,
    {"code_paths": ["linear_attn", "full_attn", "moe"]}),
("qwen3_next", {layer_types: ["full", "full"], num_local_experts: 0}, False,
    {"code_paths": ["full_attn", "dense"]}),
```

The dashboard then reports:

| Model Type | Code Paths Covered | Code Paths Missing |
|---|---|---|
| qwen3_next | linear_attn ✅, full_attn ✅, moe ✅ | dense ❌ |

### Testing the Cartesian product without explosion

Test all variants at the cheapest level, one representative at the expensive
levels:

```
L1 (graph build, <1s):       ALL variants × ALL dtypes    ← cheap, go broad
L3 (synthetic parity, ~5s):  ALL variants × f32 only      ← medium cost, still broad
L3+ (golden, minutes):       ONE representative × f32/f16 ← expensive, be selective
L4 (generation, minutes):    ONE representative × f32     ← most expensive, minimal
```

This keeps expensive tests lean while ensuring all code paths are exercised
at the broad (cheap) tiers.

### Known variant families requiring multiple configs

| Model | Variant-Triggering Fields | Current Coverage |
|---|---|---|
| qwen3_next / qwen3_5 | `layer_types` (linear vs full), MoE fields | 2 configs (mixed only) |
| deepseek_v2 | `use_mla` (MLA vs standard attn), MoE | 2 configs ✅ |
| falcon_h1 | Mamba vs attention blocks | 1 config |
| jamba / bamba | SSM + attention hybrid layers | 1 config each |
| gemma3n | `per_layer_config_type` (sliding vs global) | 1 config |
| phi3small | `gegelu_limit`, `blocksparse_params` | 1 config |
| granite | `attention_multiplier`, `logits_scaling` | 1 config |

Most need 2–3 configs to cover all paths. Total extra configs: ~10–15. At L1
cost (<1s each), this is negligible.

### VL model variant note

Vision-language models have a second variant dimension: the split format
(single model vs 3-model split vs text-only). These are typically registered
as different `model_type` entries, so the registry handles them. The 3-model
split should be treated as a deployment variant tested at L3+ (real weights
needed for split validation), not at L1/L3.

### Future: auto-discovering variant-triggering config fields

A script could inspect `__init__()` and `forward()` methods for
config-dependent branches (`if config.X`, `if self.X`, `if layer_type == Y`)
and report which config fields are variant triggers. This would warn when
`_test_configs.py` lacks configs covering all combinations — an "L0.5
meta-test" for code-path coverage completeness.

---

## 4. Dashboard Design and Communication

### Progressive disclosure

1. **Default view:** A single "Confidence" column with the highest achieved
   level. Most users just want "is this model tested?"
2. **Expanded view:** Click a model row for breakdown:
   ```
   Qwen2 (CausalLMModel)
   ├── L1 Graph Valid           ✓  (211 model types)
   ├── L2 Config Compatible     ✓  (real HF config validated)
   ├── L3 Numerically Correct   ✓  (synthetic parity: atol=2.3e-5)
   ├── L3+ Checkpoint Verified  ✓  (Qwen/Qwen2.5-0.5B, f32)
   └── L4 Generation E2E       ✓  (20-token greedy, exact match)
   ```

### Traffic-light metric display

Don't show raw numbers. Show a traffic light:

```
Qwen2.5-0.5B (f32)
  Functional:    🟢 Exact match (20/20 tokens)
  Distribution:  🟢 High fidelity (cosine=0.9999, top-10: 10/10)
  Precision:     🟢 Within tolerance (atol=2.3e-5)

Qwen2.5-0.5B-GPTQ-Int4
  Functional:    🟡 Near match (18/20 tokens)
  Distribution:  🟢 Acceptable (cosine=0.97, top-10: 7/10)
  Precision:     ⚪ N/A for quantized models
```

### Confidence breadth (not just depth)

A model at L4 tested with one prompt on one dtype may be less trustworthy
than a model at L3 tested across 5 variant configs. The dashboard should show
**both level and breadth**:

```
| Model      | Level | Variants | Dtypes | Prompts | Confidence |
| qwen3_next | L3    | 3/5      | 1/3    | 2       | ★★★☆☆      |
| llama      | L4    | 1/1      | 3/3    | 5       | ★★★★★      |
```

### The dashboard as a trust signal

The dashboard is not just internal tooling — it is a trust signal for
potential users of the library:

- **Hero metric:** "247 of 273 model types verified" (with progress bar)
- **Search-first:** Most users arrive looking for a specific model
- **Link to source:** Every row links to the test case and HF model page
- **Registry is the source of truth.** Adding a registry entry automatically
  surfaces the variant on the dashboard at L0. No separate config needed.

---

## 5. Developer Experience and CI Strategy

### The contributor workflow

1. Create model class → L1 passes automatically (existing infra)
2. Run `pytest -k "my_model" --synthetic-parity` → L3 passes locally in
   <10 seconds with no downloads
3. Register in `_registry.py` → dashboard shows L0 automatically
4. Add `test_model_id` to registry → L2 passes on nightly
5. Add YAML test case + generate golden files → L3+/L4 follow

L3 (synthetic parity) should be the **primary development loop test** —
the test contributors run 50 times while debugging. It must be fast (<5s),
deterministic (same seed → same result), and comprehensive (tests prefill +
one decode step).

### CI timing targets

| What happened | Time to know | How |
|---|---|---|
| Graph broke | <2 min | L1 in PR CI |
| Math is wrong | <5 min | L3 synthetic parity in PR CI |
| Golden file regressed | <5 min | Golden comparison in PR CI (no downloads) |
| Real model broken | Nightly | L3+ on schedule |

### Actionable failure messages

Every test failure should tell the developer **what to do next**:

```
FAILED: qwen3_moe prefill parity
  Top-1 token mismatch at position 4:
    Expected: token 3421 ("of")
    Got:      token 8912 ("the")

  → This suggests a bug in the MoE routing logic.
  → Run locally: pytest tests/e2e_golden_test.py -k "qwen3_moe" -sv
  → Compare with HF: python scripts/debug_parity.py --model Qwen/Qwen3-MoE --layer 0
```

### Variant discovery nudges

When a contributor adds a new variant, the system should guide them:

```
$ python -m pytest tests/build_graph_test.py -k "qwen3_moe"
PASSED: Graph builds successfully

Dashboard preview:
  Qwen3 family: qwen3 (L4), qwen3_moe (L1 — NEW)
  ⚠️ New variant detected. Consider adding:
    - test_model_id in registry for L2
    - Test case YAML for L3+/L4
```

---

## 6. Infrastructure Concerns

### Golden file staleness

Weekly golden regeneration creates a maintenance treadmill. Every HF
Transformers update can change tokenization or model behavior, potentially
updating 150+ files per week.

**Mitigation strategy:** Use golden files only for expensive tests (L3+/L4
with real weights). Use live comparison for cheap tests (L3 synthetic parity).
This eliminates golden file maintenance for the broad coverage layer, where
staleness risk is highest.

### Cross-platform reproducibility

Golden files generated on CPU x86-64 may not match GPU inference. Options:

- Store multiple reference points (CPU always, GPU when available).
- Establish a "platform tolerance band" — the maximum observed divergence
  between platforms.
- GPU tests pass as long as they are within the platform tolerance band, even
  if they don't exactly match the CPU golden file.

### Model removal handling

The plan covers adding models but not removing them. If a `model_type` is
removed from the registry, its golden files, test configs, and dashboard
entry become orphaned. The dashboard should detect and flag orphaned test
data.

---

## 7. Summary of Recommendations

| Area | Recommendation |
|---|---|
| **Testing tiers** | Add L3 (Synthetic Parity) as a new broad tier covering all 273 models. Rename real-weight parity to L3+ (Checkpoint Verified). Keep L4 as Generation E2E. |
| **Priority** | Ship L3 synthetic parity first — it gives parity confidence for all models immediately. Golden file infrastructure (L3+) can follow. |
| **Parity metrics** | Replace `assert_allclose` with severity-tiered evaluation: argmax agreement (gate) → top-K Jaccard (quality) → cosine similarity / KL divergence (distribution) → raw tolerance (debug). |
| **Per-dtype thresholds** | Strict for f32 (exact argmax, cosine ≥ 0.9999), relaxed for quantized (≤ 2 argmax mismatches per 20 tokens, cosine ≥ 0.95). |
| **Near-tied logits** | Flag as "ambiguous" rather than "fail" when the top-1/top-2 gap is below a threshold. |
| **Variant coverage** | Test all variants at L1/L3 (cheap), one representative at L3+/L4 (expensive). Add code-path tags to test configs for measurable coverage. |
| **Dashboard grouping** | Auto-derive families from registry prefix heuristic. Family confidence = min(variant confidences). Flag incomplete families. |
| **Dashboard confidence** | Show both level and breadth (variants × dtypes × prompts). A composite score is more honest than a simple L0–L4 label. |
| **Developer experience** | L3 synthetic parity as the primary dev loop (<5s, no network, deterministic). Progressive nudges for higher coverage levels. |
| **Golden file strategy** | Live comparison for L3 (no staleness). Golden files only for L3+/L4. Store platform tolerance bands for cross-platform reproducibility. |
| **Registry as source of truth** | Adding a registry entry automatically surfaces the variant on the dashboard at L0. No separate config files to maintain. |

### Open questions

- Should synthetic parity require `transformers` as a hard CI dependency?
  (Already a dev dependency, so probably yes.)
- Should the multi-metric evaluation be a strict hierarchy (argmax must pass
  before checking top-K) or should all metrics always be reported?
- How many random seeds is enough for synthetic parity stability? (Proposal:
  3 seeds.)
- Should code-path tags be manually maintained or auto-discovered via static
  analysis?
