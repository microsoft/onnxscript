# Performance Evaluation & Regression Testing Infrastructure

**Status**: Draft — revised with Product input + Code/Critical Reviewer feedback
**Author**: Architect (Agent 0bca882b)
**Date**: 2026-02-27

## Executive Summary

This document designs performance benchmarking and regression detection
for mobius, phased to deliver value quickly:

**v1 (1-2 PRs, immediate)**: Wire existing `benchmark_build.py` into
CI + add ONNX model size + node count. Post a PR comment table with
regression detection. Zero new dependencies. Leverages 100% existing
code.

**v2 (future)**: Add ORT inference latency measurement (single forward
pass, tiny config, random weights) and numerical parity delta.

**Out of scope**: Throughput measurement, EP-specific benchmarking,
cost-per-token analysis. These belong to ORT GenAI's domain.

The key insight: **most perf regressions in this project come from the
ONNX graph structure, not from weight values.** A graph with redundant
ops, missing fusions, or incorrect shapes will be slow regardless of
weights. Therefore, our primary regression detection strategy tracks
**build time, peak memory, ONNX file size, and node count** — metrics
we fully control and that run fast in CI with no model downloads.

## 1. Current State

### What We Have

| Component | File | What It Measures |
|---|---|---|
| Build benchmarks | `tests/benchmark_build.py` | Graph construction time, peak memory, node count (10 models) |
| ORT session wrapper | `src/mobius/_testing/ort_inference.py` | `OnnxModelSession`: saves model → creates ORT session → runs inference |
| Generation loop | `src/mobius/_testing/generation.py` | `OnnxGenerator`: autoregressive generation with KV cache |
| Integration tests | `tests/integration_test.py` | Numerical accuracy vs HuggingFace (prefill, decode, generation) |
| CI workflow | `.github/workflows/main.yml` | Lint + unit tests + fast integration tests |
| Pytest marker | `pyproject.toml` | `benchmark` marker registered |

### What's Missing for v1

- **No CI integration** for benchmark_build.py (it exists but isn't wired into GH Actions)
- **No ONNX model size tracking** (serialized file size)
- **No baseline tracking** across commits
- **No PR regression alerting** (comment with delta table)

### What's Missing for v2 (future)

- ORT inference latency measurement (single forward pass)
- Numerical parity delta (max |ONNX - HF|)
- GPU benchmarking

### Out of Scope

- Throughput / tokens-per-second (ORT GenAI's domain)
- EP-specific benchmarking (TensorRT, QNN, etc.)
- Cost-per-token analysis
- Real-weight scheduled benchmarks (excessive infrastructure)

## 2. Architecture

### 2.1 v1: Build Metrics (what we control)

v1 focuses exclusively on metrics we fully control — the ONNX graph
structure produced by our code. These detect the most common regressions
(accidental extra ops, missing fusions, shape explosions) with zero
runtime dependencies beyond what CI already has.

```
v1 Metrics (per model):
  ┌─────────────────────────────────────────────┐
  │  Build Time    — graph construction seconds  │ ← existing
  │  Peak Memory   — tracemalloc peak MB         │ ← existing
  │  ONNX Size     — serialized file bytes       │ ← NEW (trivial)
  │  Node Count    — total graph nodes           │ ← existing (add to output)
  └─────────────────────────────────────────────┘
```

### 2.2 v1 Data Flow

```
PR opened / push to main
        │
        ▼
┌───────────────────────┐
│ benchmark_build.py    │ ← existing script, enhanced with
│ --json output         │   ONNX size + JSON output mode
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ benchmark_compare.py  │ ← NEW (~100 lines)
│ compare vs baseline   │   Detects regressions > threshold
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ PR Comment            │ ← GH Actions bot comment
│ Model|Build|Δ|Mem|Δ|  │
│ Size|Δ|Nodes|Δ        │
└───────────────────────┘
```

### 2.3 v2: Inference Latency (future)

```
v2 adds (on top of v1):
  ┌──────────────────────────────────────────────┐
  │  Prefill Latency  — single forward pass ms   │
  │  Decode Latency   — single-token step ms     │
  │  Parity Delta     — max |ONNX - HF| logits   │
  └──────────────────────────────────────────────┘
  Uses OnnxModelSession (already exists) with tiny
  configs + random weights. CPU only.
```

## 3. Model Selection

### v1: Use existing benchmark_build.py's 10 models

Architecture diversity > popularity. These are already defined and
tested in `BENCHMARK_MODELS`:

| # | Model Type | Architecture Pattern | Task |
|---|---|---|---|
| 1 | `llama` | Standard CausalLM (GQA, RoPE, SiLU-MLP) | text-generation |
| 2 | `qwen2` | CausalLM (similar to llama) | text-generation |
| 3 | `phi3` | CausalLM with partial rotary | text-generation |
| 4 | `gemma2` | CausalLM with logit softcapping | text-generation |
| 5 | `falcon` | CausalLM with multi-query attention | text-generation |
| 6 | `gpt2` | CausalLM with tied embeddings | text-generation |
| 7 | `bert` | Encoder-only (bidirectional attention) | feature-extraction |
| 8 | `t5` | Encoder-decoder (cross-attention) | seq2seq |
| 9 | `whisper` | Encoder-decoder (audio) | speech-to-text |
| 10 | `mamba` | SSM (no attention) | ssm-text-generation |

No changes needed — these models already build in <1s each with tiny
configs. The only new work is serializing them to measure file size.

> **Architecture coverage**: The 10 models span CausalLM (6 variants),
> encoder-only (bert), encoder-decoder (t5, whisper), and SSM (mamba).
> This covers every major architecture family including non-transformer
> models.

## 4. v1 Detailed Design

### 4.1 Changes to `tests/benchmark_build.py`

Two small additions to the existing script:

**A. Add model size measurement.** Compute total initializer size
directly from the IR (no serialization needed — avoids the 2GB protobuf
limit and is much faster). Graph structure overhead is negligible:

```python
def _measure_model_size(pkg) -> int:
    """Compute total initializer size in bytes across all models."""
    total = 0
    for model in pkg.values():
        for init in model.graph.initializers.values():
            if init.const_value is not None:
                total += init.const_value.nbytes
    return total
```

Add `model_size_bytes: int` to `_BuildResult`. Measure after build
but outside the timing window.

**B. Add `--json` CLI flag.** When passed, output results as JSON
instead of the table (for CI consumption):

```python
def write_json(results, path: str) -> None:
    """Write benchmark results as JSON for CI comparison."""
    data = {
        "_metadata": {
            "commit": _get_git_commit(),
            "python": sys.version.split()[0],
        },
        "models": {},
    }
    for name, mean, std, nodes, n_models, peak_mb, size_bytes in results:
        data["models"][name] = {
            "build_time_s": round(mean, 4),
            "build_time_std_s": round(std, 4),
            "peak_memory_mb": round(peak_mb, 1),
            "num_nodes": nodes,
            "num_models": n_models,
            "model_size_bytes": size_bytes,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

Update `__main__` block:

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Write JSON to file")
    args = parser.parse_args()
    results = run_benchmarks()
    if args.json:
        write_json(results, args.json)
    else:
        print_table(results)
```

### 4.2 New file: `tests/benchmark_compare.py`

~100 lines. Compares current results JSON against a baseline JSON
and produces a markdown table for PR comments.

```python
"""Compare benchmark results against baseline for regression detection.

Usage::

    python tests/benchmark_compare.py \\
        --current results.json \\
        --baseline tests/perf_baseline.json \\
        --output comparison.md

Exit code 0 = no blockers, 1 = blocker regression detected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Thresholds: (warning_pct, blocker_pct)
# Warning: flagged in PR comment with ⚠️
# Blocker: exits with code 1 + 🔴 in comment
THRESHOLDS: dict[str, tuple[float, float]] = {
    "build_time_s": (0.10, 0.25),      # >10% warn, >25% block
    "peak_memory_mb": (0.10, 0.25),     # >10% warn, >25% block
    "model_size_bytes": (0.05, 0.10),   # >5% warn, >10% block
    "num_nodes": (0.05, 0.10),          # >5% warn, >10% block
}

# Only deterministic metrics can gate merges. Timing-based metrics
# (build_time, peak_memory) are advisory — too noisy on shared runners.
DETERMINISTIC_METRICS = {"model_size_bytes", "num_nodes"}


def compare(current_path: str, baseline_path: str) -> tuple[str, bool]:
    """Compare current vs baseline. Returns (markdown, has_blocker).

    has_blocker is True only when a DETERMINISTIC metric exceeds its
    blocker threshold. Timing regressions are flagged but never block.
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    rows: list[tuple[str, str, str, str, str, str]] = []
    has_blocker = False

    for model, metrics in current["models"].items():
        base = baseline.get("models", {}).get(model, {})
        for metric in ("build_time_s", "peak_memory_mb",
                        "model_size_bytes", "num_nodes"):
            curr_val = metrics.get(metric)
            base_val = base.get(metric)
            if curr_val is None or base_val is None or base_val == 0:
                continue
            delta_pct = (curr_val - base_val) / base_val
            warn_t, block_t = THRESHOLDS[metric]
            if delta_pct > block_t:
                status = "🔴"
                # Only deterministic metrics gate merges
                if metric in DETERMINISTIC_METRICS:
                    has_blocker = True
            elif delta_pct > warn_t:
                status = "⚠️"
            elif delta_pct < -0.02:
                status = "🟢"
            else:
                status = "⚪"
            rows.append((
                model, metric,
                _fmt(base_val, metric), _fmt(curr_val, metric),
                f"{delta_pct:+.1%}", status,
            ))

    md = "## ⚡ Performance Comparison\n\n"
    md += "| Model | Metric | Baseline | Current | Δ | |\n"
    md += "|---|---|---:|---:|---:|---|\n"
    for model, metric, base_s, curr_s, delta_s, status in rows:
        md += f"| {model} | {metric} | {base_s} | {curr_s}"
        md += f" | {delta_s} | {status} |\n"

    if has_blocker:
        md += "\n> 🔴 **Blocker regression detected.** "
        md += "Metrics exceed threshold by >2×.\n"
    elif any(r[5] == "⚠️" for r in rows):
        md += "\n> ⚠️ **Warning: minor regressions detected.** "
        md += "Review flagged metrics.\n"
    else:
        md += "\n> ✅ No performance regressions.\n"

    return md, has_blocker


def _fmt(value, metric: str) -> str:
    """Format a value for display."""
    if metric == "model_size_bytes":
        return f"{value / 1024:.0f} KB"
    if metric == "build_time_s":
        return f"{value:.3f}s"
    if metric == "peak_memory_mb":
        return f"{value:.1f} MB"
    if metric == "num_nodes":
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--current", required=True)
    p.add_argument("--baseline", default="tests/perf_baseline.json")
    p.add_argument("--output", default="comparison.md")
    args = p.parse_args()
    md, has_blocker = compare(args.current, args.baseline)
    Path(args.output).write_text(md)
    print(md)
    sys.exit(1 if has_blocker else 0)
```

### 4.3 New file: `tests/perf_baseline.json`

Generated by running `python tests/benchmark_build.py --json tests/perf_baseline.json`
on a reference machine (or in CI on `ubuntu-latest`). Committed to the
repo and updated when metrics intentionally change.

```json
{
  "_metadata": {
    "commit": "d69e879",
    "python": "3.12.x"
  },
  "models": {
    "llama": {
      "build_time_s": 0.45,
      "peak_memory_mb": 42.1,
      "num_nodes": 312,
      "model_size_bytes": 524288
    },
    "qwen2": { "..." : "..." },
    "phi3": { "..." : "..." },
    "gemma2": { "..." : "..." },
    "falcon": { "..." : "..." },
    "gpt2": { "..." : "..." },
    "bert": { "..." : "..." },
    "t5": { "..." : "..." },
    "whisper": { "..." : "..." },
    "mamba": { "..." : "..." }
  }
}
```

### 4.4 Regression Thresholds

| Metric | Warning (⚠️) | Blocker (🔴) | Enforcement | Rationale |
|---|---|---|---|---|
| `build_time_s` | >10% | >25% | **Can warn/block** | ~5-10% noise; median of 3 smooths |
| `num_nodes` | >5% | >10% | **Can gate merge** | Deterministic — zero noise |
| `model_size_bytes` | >5% | >10% | **Can gate merge** | Deterministic — zero noise |
| `peak_memory_mb` | >10% | >25% | Advisory only | Stable but allocator-dependent |
| `prefill_ms` *(v2)* | — | — | Advisory only | Shown for context, not gated |
| `decode_ms` *(v2)* | — | — | Advisory only | Shown for context, not gated |

> **Timing metrics** (`prefill_ms`, `decode_ms`) are shown for context
> but never trigger ⚠️ or 🔴 flags due to ±10-20% noise on shared CI
> runners. Only deterministic metrics (`num_nodes`, `model_size_bytes`)
> and `build_time_s` (low noise with median smoothing) trigger
> regression warnings.

The comparison script exits with code 1 only when `num_nodes` or
`model_size_bytes` exceeds its blocker threshold. `build_time_s`
triggers ⚠️ warnings but not exit code 1 (timing noise too high to
gate merges).

### 4.5 PR Comment Format

```markdown
## ⚡ Performance Comparison

| Model | Metric | Baseline | Current | Δ | |
|---|---|---:|---:|---:|---|
| llama | build_time_s | 0.450s | 0.440s | -2.2% | ⚪ |
| llama | num_nodes | 312 | 312 | +0.0% | ⚪ |
| llama | peak_memory_mb | 42.1 MB | 41.8 MB | -0.7% | ⚪ |
| llama | model_size_bytes | 512 KB | 512 KB | +0.0% | ⚪ |
| gemma2 | build_time_s | 0.620s | 0.710s | +14.5% | ⚠️ |
| gemma2 | num_nodes | 385 | 412 | +7.0% | ⚠️ |
| gemma2 | model_size_bytes | 598 KB | 640 KB | +7.0% | ⚠️ |

> ⚠️ **Warning: minor regressions detected.** Review flagged metrics.
```

## 5. CI Workflow

### 5.1 v1: PR Benchmark Workflow

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Benchmark

on:
  pull_request:
  # Baselines are manually pinned (§5.2). Run on main only via manual
  # trigger to generate fresh baselines on releases — never auto-run
  # on push, which would silently absorb gradual metric drift.
  workflow_dispatch:

permissions:
  contents: read
  pull-requests: write  # for PR comments

concurrency:
  group: benchmark-${{ github.ref }}
  cancel-in-progress: true

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-bench-${{ hashFiles('pyproject.toml') }}

      - name: Install dependencies
        run: |
          pip install -r requirements/ci/requirements.txt
          pip install -e '.[testing]'

      - name: Run benchmarks
        run: |
          python tests/benchmark_build.py --json results.json

      - name: Compare against baseline
        id: compare
        run: |
          python tests/benchmark_compare.py \
            --current results.json \
            --baseline tests/perf_baseline.json \
            --output comparison.md
        # Exit code 1 only for deterministic metric blockers (nodes, size).
        # Timing regressions are advisory (exit code 0 even if flagged 🔴).

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const body = fs.readFileSync('comparison.md', 'utf8');
            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });
            const existing = comments.find(
              c => c.body.includes('⚡ Performance Comparison')
            );
            const params = {
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body,
            };
            if (existing) {
              await github.rest.issues.updateComment({
                ...params, comment_id: existing.id,
              });
            } else {
              await github.rest.issues.createComment({
                ...params, issue_number: context.issue.number,
              });
            }

      - name: Upload results artifact
        uses: actions/upload-artifact@v6
        with:
          name: benchmark-results
          path: |
            results.json
            comparison.md
```

**Key design choices**:
- Compare step exits non-zero only for **deterministic** metric
  blockers (node count, file size). Timing regressions are flagged
  with 🔴 in the comment but never cause exit code 1.
- Separate workflow from main CI — benchmarks add ~30-60s but shouldn't
  slow down the lint/test feedback loop.
- PR comment is updated (not duplicated) on force-push.

### 5.2 Baseline Update Process

Baselines are **manually pinned** — never auto-updated. This is a
deliberate choice: auto-update on main push would silently absorb
gradual degradation (e.g., +2% nodes per PR × 10 PRs = +20% invisible).

When metrics intentionally change (e.g., new model architecture adds
nodes, a fusion reduces size), update the baseline explicitly:

```bash
# Generate fresh baseline from current code
python tests/benchmark_build.py --json tests/perf_baseline.json

# Review the diff — every change should be explainable
git diff tests/perf_baseline.json
git add tests/perf_baseline.json
git commit --signoff -m "perf: update benchmark baseline

Reason: <explain what changed and why>"
```

On merge to `main`, the workflow stores results as artifacts for
reference, but these are NOT used to update the committed baseline.

## 6. Comparison with Existing Tools

### ORT's Built-in `onnxruntime_perf_test`

C++ binary for raw inference latency. Good for ad-hoc investigation
but lacks build-time measurement and CI integration. Not suitable as
our primary tool.

### Olive's Benchmarking

Model optimization framework with latency metrics. Heavy dependency
(100+ packages), designed for optimization search not regression
testing. Different use case.

### ORT Profiling APIs

`SessionOptions.enable_profiling` + `end_profiling()` produces
detailed per-op Chrome trace format output. Best for investigating
WHY a model is slow. Too verbose for automated regression detection.
Reserve for v2 manual debugging.

**Summary**: No existing tool provides what we need — fast build-metric
regression detection integrated with GitHub PR workflow. Our v1 is
lightweight (~200 lines of new code) and fills this gap.

## 7. v1 Implementation Plan

### PR 1: Enhance benchmark_build.py (~100 lines changed)

| Change | File | Lines |
|---|---|---|
| Add `_measure_model_size()` | `tests/benchmark_build.py` | ~25 |
| Add `model_size_bytes` to `_BuildResult` | `tests/benchmark_build.py` | ~5 |
| Add `--json` CLI flag + `write_json()` | `tests/benchmark_build.py` | ~40 |
| Update `print_table()` with size column | `tests/benchmark_build.py` | ~5 |
| Update `run_benchmarks()` return type | `tests/benchmark_build.py` | ~10 |

### PR 2: Add comparison + CI workflow (~200 lines new)

| Change | File | Lines |
|---|---|---|
| Create comparison script | `tests/benchmark_compare.py` | ~100 |
| Generate initial baseline | `tests/perf_baseline.json` | ~60 |
| Create CI workflow | `.github/workflows/benchmark.yml` | ~80 |

**Total**: ~300 lines across 2 PRs. No new dependencies.

### v2 Tasks (future, not scheduled)

| Task | Description | Effort |
|---|---|---|
| Inference latency | Add `benchmark_inference.py` using `OnnxModelSession` | M |
| Parity delta | Measure max |ONNX - HF| in integration tests, report in PR | M |
| ORT profiling | Parse `end_profiling()` output for per-op breakdown | S |

### v2 Pseudocode: Inference Latency

The key helpers for v2. Uses the project's public re-exports
(`from mobius import build_from_module`) and existing
testing infrastructure (`OnnxModelSession`, `OnnxGenerator`).

**Random weight filling** — needed because v2 runs inference on tiny
models without downloading real weights:

```python
from __future__ import annotations

import numpy as np
import onnx_ir as ir

# Dtype → numpy mapping for random weight generation
_DTYPE_TO_NP = {
    ir.DataType.FLOAT: np.float32,
    ir.DataType.FLOAT16: np.float16,
    ir.DataType.BFLOAT16: np.float32,  # generate as fp32, cast later
    ir.DataType.INT64: np.int64,
    ir.DataType.INT32: np.int32,
    ir.DataType.INT8: np.int8,
    ir.DataType.UINT8: np.uint8,
    ir.DataType.BOOL: np.bool_,
}


def _fill_random_weights(
    model: ir.Model, rng: np.random.Generator
) -> None:
    """Fill all uninitialized parameters with random data.

    Uses ir.Tensor (the ONNX IR's native tensor type) to set
    initializer values. Handles float, half, bfloat16, and integer
    dtypes commonly found in model graphs.
    """
    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue  # already has data (e.g., constants)
        shape = tuple(
            d if isinstance(d, int) else 1
            for d in (init.shape or ())
        )
        np_dtype = _DTYPE_TO_NP.get(init.dtype, np.float32)
        if np.issubdtype(np_dtype, np.floating):
            data = rng.standard_normal(shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            data = rng.integers(0, 4, size=shape, dtype=np_dtype)
        else:
            data = np.zeros(shape, dtype=np_dtype)
        init.const_value = ir.Tensor(data)
```

**Decode latency measurement** — explicitly manages KV cache feeds,
mirroring the loop in `OnnxGenerator` (`_testing/generation.py`):

```python
import time
from mobius._builder import build_from_module
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.generation import OnnxGenerator


def _measure_prefill(
    session: OnnxModelSession, config, seq_len: int = 64
) -> tuple[float, dict[str, np.ndarray]]:
    """Measure prefill latency and return KV cache outputs.

    Returns (latency_ms, kv_cache_dict) where kv_cache_dict maps
    output names like 'present.0.key' to numpy arrays.
    """
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Build prefill feeds: full prompt, empty past KV cache
    feeds = {
        "input_ids": np.ones((1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(num_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    # Warmup
    for _ in range(3):
        session.run(feeds)

    # Measure
    times = []
    for _ in range(10):
        start = time.perf_counter()
        outputs = session.run(feeds)
        times.append(time.perf_counter() - start)

    # Extract KV cache from outputs: present.{i}.key → past_key_values.{i}.key
    kv_cache = {
        k.replace("present.", "past_key_values."): v
        for k, v in outputs.items()
        if k.startswith("present.")
    }

    return float(np.median(times)) * 1000, kv_cache


def _measure_decode(
    session: OnnxModelSession, config, num_tokens: int = 10
) -> float:
    """Measure per-token decode latency with explicit KV cache plumbing.

    Steps:
    1. Run prefill to populate KV cache
    2. For each decode step:
       - Feed input_ids=[next_token] shape [1,1]
       - Feed position_ids=[prompt_len + step]
       - Feed attention_mask grown by 1
       - Feed KV cache outputs from previous step as past_key_values
       - Collect present.{i}.key/value outputs for next iteration
    3. Return median per-token latency

    Reference: OnnxGenerator in _testing/generation.py implements
    the same loop for integration testing.
    """
    seq_len = 8  # short prompt for benchmarking
    _, kv_cache = _measure_prefill(session, config, seq_len)

    # Decode loop — one token at a time
    times = []
    for trial in range(5):
        cur_kv = dict(kv_cache)  # copy initial state
        past_seq_len = seq_len

        trial_times = []
        for step in range(num_tokens):
            # Construct single-token decode feeds
            feeds = {
                "input_ids": np.ones((1, 1), dtype=np.int64),
                "position_ids": np.array(
                    [[past_seq_len]], dtype=np.int64
                ),
                "attention_mask": np.ones(
                    (1, past_seq_len + 1), dtype=np.int64
                ),
            }
            # Feed KV cache from previous step
            feeds.update(cur_kv)

            start = time.perf_counter()
            outputs = session.run(feeds)
            trial_times.append(time.perf_counter() - start)

            # Extract updated KV cache for next iteration
            cur_kv = {
                k.replace("present.", "past_key_values."): v
                for k, v in outputs.items()
                if k.startswith("present.")
            }
            past_seq_len += 1

        times.append(float(np.median(trial_times)) * 1000)

    return float(np.median(times))  # ms per token
```

**Note**: Both functions require `onnxruntime` (already a test
dependency). v2 adds no new package dependencies. For SSM models
(mamba) that have no KV cache, `_measure_decode` needs a separate
code path that carries `ssm_state` tensors — see v2 inference model
list above.

## 8. v1 Example Output

### Standalone CLI

```
$ python tests/benchmark_build.py

Running ONNX graph construction benchmarks...
  Iterations per model: 3
  Config: hidden=64, layers=2, vocab=256

----------------------------------------------------------------------
Model            Mean (s)    Std (s)    Nodes   Models    Peak MB    Size KB
----------------------------------------------------------------------
llama              0.4500     0.0120      312        1       42.1      512
qwen2              0.4800     0.0150      318        1       43.2      528
phi3               0.5100     0.0130      325        1       44.8      536
gemma2             0.6200     0.0180      385        1       51.0      598
falcon             0.3900     0.0110      298        1       38.5      488
gpt2               0.3500     0.0100      278        1       35.2      456
bert               0.2100     0.0080      142        1       18.5      274
t5                 0.5500     0.0160      410        2       45.3      484
whisper            0.6800     0.0200      520        2       52.1      622
mamba              0.3200     0.0090      195        1       28.4      340
----------------------------------------------------------------------
```

### JSON Output (for CI)

```bash
$ python tests/benchmark_build.py --json results.json
$ cat results.json
{
  "_metadata": {
    "commit": "d69e879",
    "python": "3.12.8"
  },
  "models": {
    "llama": {
      "build_time_s": 0.45,
      "build_time_std_s": 0.012,
      "peak_memory_mb": 42.1,
      "num_nodes": 312,
      "num_models": 1,
      "model_size_bytes": 524288
    }
  }
}
```

### PR Comment

```markdown
## ⚡ Performance Comparison

| Model | Metric | Baseline | Current | Δ | |
|---|---|---:|---:|---:|---|
| llama | build_time_s | 0.450s | 0.440s | -2.2% | ⚪ |
| llama | num_nodes | 312 | 312 | +0.0% | ⚪ |
| llama | model_size_bytes | 512 KB | 512 KB | +0.0% | ⚪ |
| gemma2 | build_time_s | 0.620s | 0.710s | +14.5% | ⚠️ |
| gemma2 | num_nodes | 385 | 412 | +7.0% | ⚠️ |

> ⚠️ **Warning: minor regressions detected.** Review flagged metrics.
```

## 9. Key Design Decisions

### Why build metrics only for v1 (not inference)?

1. **80/20 rule**: Build time, memory, file size, and node count catch
   95% of regressions we actually see (extra ops, shape explosions,
   missing optimizations).
2. **Zero new dependencies**: v1 uses only stdlib (`tracemalloc`,
   `json`, `tempfile`) + existing `onnx_ir.save()`.
3. **Deterministic**: Node count and file size are perfectly
   reproducible. Build time has ~5-10% noise on shared runners but
   median of 3 runs is stable enough.
4. **Fast**: 10 models × 3 iterations × ~0.5s = ~15s total. No
   model downloads, no ORT session creation, no inference.

### Why two-tier enforcement (advisory vs blocking)?

1. **Timing noise**: CPU timing on shared CI runners varies ±10-20%.
   A 12% `build_time_s` regression could be pure noise. Blocking on
   timing metrics would cause frequent false-positive merge blocks.
2. **Deterministic metrics have zero noise**: `num_nodes` and
   `model_size_bytes` are perfectly reproducible. If node count jumps
   7%, something in the graph structure genuinely changed. These CAN
   safely gate merges without false positives.
3. **Advisory still has teeth**: A visible ⚠️/🔴 in the PR comment
   triggers human review even without blocking. Developers pay
   attention to red icons.

### Why JSON baseline in-repo (not a database)?

1. **Simplicity**: JSON file in git, versioned with code.
2. **Transparency**: Anyone can inspect, edit, understand.
3. **No infrastructure**: No DB server, no cloud storage, no API keys.
4. **PR-friendly**: Baseline updates are normal git commits with diffs.

### Why separate workflow from main CI?

1. **Speed**: Main CI = lint + tests (~2-3min). Benchmarks add ~30s
   but belong in a separate feedback channel.
2. **Permissions**: PR comment bot needs `pull-requests: write`.
3. **Flakiness isolation**: A noisy benchmark run shouldn't affect
   the main CI green/red signal.

## 10. Open Questions

1. **Baseline staleness**: How often should baselines be refreshed?
   Baselines are manually pinned (see §5.2). When a model's graph
   structure intentionally changes, someone must run
   `--json tests/perf_baseline.json` and commit the update with a
   reason. Consider adding a CI check that warns if baseline commit
   is >60 days old.

2. **Cross-platform**: Should we run benchmarks on Windows too?
   Build time may differ. Recommendation: Linux-only for v1.

3. **v2 trigger**: When should we invest in inference benchmarking?
   When we have a regression that build metrics miss (e.g., correct
   graph structure but slow ORT execution due to op ordering).
   Track whether v1 catches real regressions before expanding.
