# Golden File Infrastructure Design — L4 & L5 Testing

**Author:** Architect Agent (bef97f8c)
**Date:** 2026-03-06
**Status:** Design proposal — ready for review

---

## Executive Summary

This document specifies the golden file infrastructure for **L4 (Checkpoint Verified)** and **L5 (Generation E2E)** testing layers. The core design principle: **adding test coverage = adding a YAML file**. No code changes needed.

The system has 4 components:
1. **Test case YAML files** — declarative specifications of what to test
2. **Golden reference data** — pre-computed HuggingFace outputs stored as `.npz` files
3. **`golden.py` API** — load/save/discover functions for test infrastructure
4. **`generate_golden.py` CLI** — offline tool to (re)generate golden reference data

### Key Design Decision: Challenge the Problem Framing

The prior `compare_golden()` API accepts only `top1_id, top2_id, top10_ids` — extremely minimal golden data. This is intentional for L4 (argmax-gated), but **insufficient for L5 (generation) and for debugging L4 failures**.

**My recommendation:** Store richer golden data than the minimum gate requires. The cost is ~2KB per test case (not full vocab logits). The benefit: when a test fails, you can immediately diagnose *why* without re-running HuggingFace. The golden file becomes a **debugging artifact**, not just a pass/fail oracle.

---

## 1. YAML Test Case Schema

Each test case is a single YAML file describing one model + one input scenario. The schema is intentionally flat — no nesting beyond what's structurally necessary.

### Schema Definition

```yaml
# --- Required fields ---
model_id: str            # HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B")
revision: str            # Git commit SHA or tag (e.g., "abc123" or "main")
task_type: str           # One of the task strings from _registry.py
dtype: str               # "float32", "float16", "bfloat16"

# --- Inputs (at least one must be present) ---
inputs:
  prompts: list[str]     # Text prompts (causal-lm, seq2seq, VL)
  images: list[str]      # Paths relative to testdata/ (VL, vision)
  audio: list[str]       # Paths relative to testdata/ (audio, speech)
  decoder_prompt: str    # Optional: forced decoder prefix (seq2seq, whisper)

# --- Generation params (for L5 only; omit for L4-only cases) ---
generation:
  max_new_tokens: int    # Default: 20
  do_sample: false       # ALWAYS false for golden tests (greedy only)
  temperature: 1.0       # Fixed at 1.0 for reproducibility

# --- Expected output type (determines what golden data contains) ---
level: str               # "L4", "L5", or "L4+L5" (both)

# --- Optional metadata ---
trust_remote_code: bool  # Default: false
notes: str               # Human-readable description
skip_reason: str         # If present, test is skipped with this message
```

### Concrete Examples

#### Example 1: Causal LM (Qwen2.5) — L4+L5

```yaml
# testdata/cases/causal-lm/qwen2_5-0_5b.yaml
model_id: "Qwen/Qwen2.5-0.5B"
revision: "a8e65d8c71e8e6b3b4a4e5f7b2c1d0e9f3a6b8c7"
task_type: "text-generation"
dtype: "float32"

inputs:
  prompts:
    - "The capital of France is"

level: "L4+L5"

generation:
  max_new_tokens: 20
  do_sample: false

notes: "Representative causal-lm model, smallest Qwen2.5 variant"
```

#### Example 2: Vision-Language (Qwen2.5-VL) — L4+L5

```yaml
# testdata/cases/vision-language/qwen2_5-vl-3b.yaml
model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
revision: "d4f0e2a3b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0"
task_type: "qwen-vl"
dtype: "float32"

inputs:
  prompts:
    - "Describe this image in detail."
  images:
    - "pipeline-cat-chonk.jpeg"

level: "L4+L5"

generation:
  max_new_tokens: 30
  do_sample: false

notes: "VL model with image input. Tests vision encoder + embedding + decoder pipeline."
```

#### Example 3: Encoder-Only (BERT) — L4 only

```yaml
# testdata/cases/encoder/bert-base.yaml
model_id: "google-bert/bert-base-uncased"
revision: "86b5e0934494bd15c9632b12f734a8a67f723594"
task_type: "feature-extraction"
dtype: "float32"

inputs:
  prompts:
    - "The quick brown fox jumps over the lazy dog."

level: "L4"

notes: "Encoder-only model. No generation — L4 compares last_hidden_state argmax."
```

#### Example 4: Seq2Seq (T5) — L4+L5

```yaml
# testdata/cases/seq2seq/t5-small.yaml
model_id: "google-t5/t5-small"
revision: "df1b051c49ef4a5c8a7346b9b tried3e2f1a4b5c6d7"
task_type: "seq2seq"
dtype: "float32"

inputs:
  prompts:
    - "translate English to German: The house is wonderful."
  decoder_prompt: ""

level: "L4+L5"

generation:
  max_new_tokens: 20
  do_sample: false

notes: "Encoder-decoder model. Tests encoder + decoder forward and generation."
```

#### Example 5: Audio/Speech (Whisper) — L4+L5

```yaml
# testdata/cases/audio/whisper-tiny.yaml
model_id: "openai/whisper-tiny"
revision: "b9e6e1cc8a867e47043ee71e6da9e50ee97d69f0"
task_type: "speech-to-text"
dtype: "float32"

inputs:
  audio:
    - "652-129742-0006.flac"

level: "L4+L5"

generation:
  max_new_tokens: 50
  do_sample: false

notes: "Speech-to-text encoder-decoder. Audio file from LibriSpeech."
```

---

## 2. Golden File Format

### Recommendation: **NumPy `.npz`** (compressed)

**Rationale:**

| Format | Size (per case) | Read speed | Python native | Human-debuggable | Supports mixed types |
|--------|----------------|------------|---------------|------------------|---------------------|
| `.npz` | ~2-10 KB | <1ms | ✅ `np.load()` | ❌ (binary) | ✅ (named arrays) |
| safetensors | ~2-10 KB | <1ms | ✅ (with lib) | ❌ (binary) | ⚠️ (tensors only) |
| JSON | ~5-50 KB | ~5ms | ✅ | ✅ | ✅ |

**Why `.npz` wins:**

1. **Zero extra dependencies.** `numpy` is already a core dependency. `safetensors` would add a dependency to the test infrastructure that doesn't exist today.
2. **Named arrays.** `.npz` stores multiple named arrays in one file — perfect for `top1_id`, `top10_ids`, `logits_summary`, `generated_ids`, etc.
3. **Compact.** We're storing ~100 integers + a few floats per test case, not full vocab logits. Binary is naturally compact.
4. **Fast.** Sub-millisecond load times. With 50-100 test cases, total load time is negligible.
5. **Already used in the ecosystem.** NumPy is the interchange format throughout this codebase (`ort_inference.py` returns `dict[str, np.ndarray]`, `torch_reference.py` converts to numpy).

**Why NOT JSON:** Golden files include numpy arrays (top-10 IDs, generated token IDs). JSON requires manual serialization/deserialization of arrays and loses type information. This is friction for no benefit.

**Why NOT safetensors:** Adds a runtime dependency for test infrastructure. Safetensors is designed for large weight tensors, not small metadata arrays. Overkill here.

### What Data is Stored

**For L4 (Checkpoint Verified) — prefill forward pass:**

```python
# Arrays stored in the .npz file:
{
    # --- Gate data (required by compare_golden()) ---
    "top1_id": np.array([12366]),              # int64, shape (1,)
    "top2_id": np.array([8234]),               # int64, shape (1,)
    "top10_ids": np.array([12366, 8234, ...]),  # int64, shape (10,)

    # --- Diagnostic data (for debugging failures) ---
    "top10_logits": np.array([15.2, 14.8, ...]),  # float64, shape (10,)
    "logits_summary": np.array([                   # float64, shape (4,)
        15.2,     # max logit
        -12.3,    # min logit
        0.042,    # mean logit
        3.7,      # std logit
    ]),

    # --- Reproduction metadata ---
    "input_ids": np.array([[1, 450, 3127, ...]]),  # int64, tokenized input
}
```

**For L5 (Generation E2E) — greedy generation:**

```python
{
    # --- L4 data (all of the above) ---
    ...

    # --- L5-specific data ---
    "generated_ids": np.array([12366, 8234, 551, ...]),  # int64, shape (N,)
    "generated_text": ...  # NOT stored — derived from generated_ids + tokenizer
}
```

**Design decision: NO full-vocab logits.**

Full vocab logits for a single forward pass of a 32K-vocab model = 128KB float32. For 100 test cases, that's 12.8MB in the repo. Not catastrophic, but:
1. We don't need them — `compare_golden()` only uses argmax and top-10.
2. If we ever need full logits for debugging, `generate_golden.py --full-logits` can regenerate them locally. Don't store permanently what you can regenerate.
3. Keeping golden files small means they can live in the git repo without Git LFS.

**Size estimate:** ~2-5 KB per test case. 100 test cases = ~500 KB total. Comfortably fits in a git repo.

### Multi-Model Tasks (VL, Seq2Seq)

For tasks that produce multiple ONNX models (e.g., VisionLanguageTask → decoder, vision, embedding), store golden data **per-component** in the same `.npz`:

```python
{
    # Decoder (text) golden data — same as causal-lm
    "decoder.top1_id": np.array([12366]),
    "decoder.top10_ids": np.array([12366, 8234, ...]),
    "decoder.top10_logits": np.array([15.2, 14.8, ...]),
    "decoder.logits_summary": np.array([15.2, -12.3, 0.042, 3.7]),
    "decoder.input_ids": np.array([[1, 450, ...]]),

    # Vision encoder golden data
    "vision.output_norm": np.array([42.5]),         # L2 norm of vision output
    "vision.output_shape": np.array([1, 577, 1024]), # shape of output tensor

    # Embedding golden data
    "embedding.output_norm": np.array([38.2]),       # L2 norm
    "embedding.output_shape": np.array([1, 583, 2048]),

    # L5 generation
    "generated_ids": np.array([12366, 8234, 551, ...]),
}
```

**Rationale for vision/embedding:** We don't compare full intermediate tensors (too large, not meaningful as golden data). Instead, store shape + L2 norm as sanity checks. The real gate is the decoder's argmax — if the decoder produces the right token, the vision pipeline worked.

---

## 3. Directory Layout

```
testdata/
├── cases/                              # YAML test case definitions
│   ├── causal-lm/
│   │   ├── qwen2_5-0_5b.yaml
│   │   ├── llama-3_2-1b.yaml
│   │   ├── phi-3_5-mini.yaml
│   │   ├── gemma-3-1b.yaml
│   │   └── smollm-135m.yaml
│   ├── vision-language/
│   │   ├── qwen2_5-vl-3b.yaml
│   │   └── llava-v1_6-mistral-7b.yaml
│   ├── encoder/
│   │   ├── bert-base.yaml
│   │   └── distilbert-base.yaml
│   ├── seq2seq/
│   │   ├── t5-small.yaml
│   │   └── bart-base.yaml
│   ├── audio/
│   │   ├── whisper-tiny.yaml
│   │   └── wav2vec2-base.yaml
│   └── vision/
│       └── vit-base.yaml
│
├── golden/                             # Pre-computed HF reference outputs
│   ├── causal-lm/
│   │   ├── qwen2_5-0_5b.npz
│   │   ├── llama-3_2-1b.npz
│   │   └── ...
│   ├── vision-language/
│   │   ├── qwen2_5-vl-3b.npz
│   │   └── ...
│   ├── encoder/
│   │   └── bert-base.npz
│   ├── seq2seq/
│   │   └── t5-small.npz
│   ├── audio/
│   │   └── whisper-tiny.npz
│   └── vision/
│       └── vit-base.npz
│
├── default_tolerances.yaml             # Tolerance config (see §4)
│
├── 652-129742-0006.flac                # Existing audio fixture
├── pipeline-cat-chonk.jpeg             # Existing image fixture
└── qwen2_5_genai_config/              # Existing config fixture
```

### Naming Convention

**YAML filenames** derive from the HuggingFace model ID:
- Replace `/` with `-`
- Replace `.` with `_`
- Lowercase everything
- Examples:
  - `Qwen/Qwen2.5-0.5B` → `qwen2_5-0_5b.yaml`
  - `meta-llama/Llama-3.2-1B` → `llama-3_2-1b.yaml`
  - `google-bert/bert-base-uncased` → `bert-base-uncased.yaml`

**Golden filenames** match YAML filenames exactly (same stem, `.npz` extension).

**Why organized by task type (not model family)?**
1. Task type determines the test runner logic (what inputs to prepare, what outputs to compare). Grouping by task type means `discover_test_cases(task_type="causal-lm")` is a simple glob.
2. Model family is already captured in the registry. Duplicating it in the directory structure adds complexity without value.
3. A model can belong to one family but different task types (e.g., `Qwen2.5` is causal-lm, `Qwen2.5-VL` is vision-language). Task type is the natural grouping axis for golden tests.

---

## 4. Tolerance Config Format

```yaml
# testdata/default_tolerances.yaml
#
# Tolerance thresholds for L4 (Checkpoint Verified) and L5 (Generation E2E).
# These are defaults — individual test cases can override via YAML fields.

L4:
  # L4 gate: argmax match. These are diagnostic thresholds, not gates.
  float32:
    near_tie_margin: 0.01
    top10_jaccard_warn: 0.7       # Warn if Jaccard drops below this
    cosine_similarity_warn: 0.999  # Warn if cosine drops below this
  float16:
    near_tie_margin: 0.1
    top10_jaccard_warn: 0.5
    cosine_similarity_warn: 0.99
  bfloat16:
    near_tie_margin: 0.5
    top10_jaccard_warn: 0.4
    cosine_similarity_warn: 0.95
  int4:
    near_tie_margin: 1.0
    top10_jaccard_warn: 0.3
    cosine_similarity_warn: 0.9

L5:
  # L5 gate: exact token sequence match for greedy generation.
  # Tolerance is binary (match/no match), but we allow partial credit.
  float32:
    min_token_match_ratio: 1.0    # All tokens must match
  float16:
    min_token_match_ratio: 0.9    # Allow 10% mismatch for f16 quantization
  bfloat16:
    min_token_match_ratio: 0.8
  int4:
    min_token_match_ratio: 0.7

L3:
  # L3 (synthetic) uses atol/rtol from parity.py DEFAULT_TOLERANCES.
  # Kept here for completeness / documentation.
  float32:
    atol: 1.0e-3
    rtol: 1.0e-3
  float16:
    atol: 1.0e-1
    rtol: 1.0e-1
  bfloat16:
    atol: 5.0e-1
    rtol: 5.0e-1
```

### Design Rationale

- **L4 is argmax-gated, not atol-gated.** The tolerance config for L4 stores *diagnostic* thresholds (Jaccard, cosine). These produce warnings, not failures. The gate is still `compare_golden()`.
- **L5 uses `min_token_match_ratio`** rather than strict equality for non-f32 dtypes. Quantized models may diverge after several generation steps due to accumulated rounding. A 90% match on a 20-token sequence means ≤2 tokens differ — still a useful signal.
- **Per-model overrides** are not in this file. If a specific model needs custom tolerances, add `tolerance_overrides:` to its YAML test case. The tolerance config is the *default*, not a per-model registry.

---

## 5. `golden.py` API

File: `src/mobius/_testing/golden.py`

### Complete Module Interface

```python
"""Golden file loading, saving, and test case discovery for L4/L5 tests.

This module provides the data layer for golden-file-based testing:
- Load/save YAML test case definitions
- Load/save .npz golden reference data
- Discover test cases by task type for pytest parametrization
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


# --- Constants ---

CASES_DIR = Path("testdata/cases")
GOLDEN_DIR = Path("testdata/golden")
DEFAULT_TOLERANCES_PATH = Path("testdata/default_tolerances.yaml")


# --- Data Classes ---

@dataclasses.dataclass(frozen=True)
class TestCase:
    """A parsed YAML test case definition."""
    case_id: str              # Stem of the YAML file (e.g., "qwen2_5-0_5b")
    task_type: str            # "text-generation", "feature-extraction", etc.
    model_id: str             # HuggingFace model ID
    revision: str             # HF model revision/commit SHA
    dtype: str                # "float32", "float16", "bfloat16"
    level: str                # "L4", "L5", or "L4+L5"
    prompts: list[str]        # Text inputs (may be empty)
    images: list[str]         # Paths relative to testdata/ (may be empty)
    audio: list[str]          # Paths relative to testdata/ (may be empty)
    decoder_prompt: str       # Forced decoder prefix (may be empty)
    generation_params: dict   # {"max_new_tokens": 20, "do_sample": false, ...}
    trust_remote_code: bool
    skip_reason: str | None   # If set, test should be skipped
    yaml_path: Path           # Absolute path to the source YAML file


@dataclasses.dataclass(frozen=True)
class GoldenRef:
    """Pre-computed HuggingFace reference outputs."""
    # L4 data
    top1_id: int
    top2_id: int
    top10_ids: list[int]
    top10_logits: list[float]
    logits_summary: np.ndarray      # [max, min, mean, std]
    input_ids: np.ndarray           # Tokenized input used

    # L5 data (None if level == "L4")
    generated_ids: np.ndarray | None

    # Multi-model diagnostics (None for single-model tasks)
    component_norms: dict[str, float]    # e.g., {"vision": 42.5, "embedding": 38.2}
    component_shapes: dict[str, tuple]   # e.g., {"vision": (1, 577, 1024)}

    # Metadata
    npz_path: Path              # Absolute path to the .npz file


@dataclasses.dataclass(frozen=True)
class Tolerances:
    """Tolerance thresholds for a specific level + dtype."""
    near_tie_margin: float
    top10_jaccard_warn: float
    cosine_similarity_warn: float
    min_token_match_ratio: float    # L5 only; 1.0 for L4


# --- Public API ---

def load_test_case(yaml_path: Path) -> TestCase:
    """Load a single test case from a YAML file.

    Args:
        yaml_path: Path to the YAML test case file.

    Returns:
        Parsed TestCase dataclass.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    ...


def load_golden_ref(npz_path: Path) -> GoldenRef:
    """Load golden reference data from an .npz file.

    Args:
        npz_path: Path to the .npz golden file.

    Returns:
        Parsed GoldenRef dataclass.

    Raises:
        FileNotFoundError: If the .npz file does not exist.
    """
    ...


def save_golden_ref(
    npz_path: Path,
    *,
    top1_id: int,
    top2_id: int,
    top10_ids: list[int],
    top10_logits: list[float],
    logits_summary: np.ndarray,
    input_ids: np.ndarray,
    generated_ids: np.ndarray | None = None,
    component_norms: dict[str, float] | None = None,
    component_shapes: dict[str, tuple] | None = None,
) -> None:
    """Save golden reference data to an .npz file.

    Args:
        npz_path: Path to write the .npz file.
        top1_id: Argmax token ID from HF last-token logits.
        top2_id: Second-highest token ID.
        top10_ids: Top-10 token IDs sorted by descending logit.
        top10_logits: Corresponding logit values for top-10 tokens.
        logits_summary: [max, min, mean, std] of full logit vector.
        input_ids: Tokenized input array.
        generated_ids: Full generated token sequence (L5 only).
        component_norms: L2 norms for multi-model component outputs.
        component_shapes: Output shapes for multi-model components.
    """
    ...


def discover_test_cases(
    task_type: str | None = None,
    level: str | None = None,
    root: Path = CASES_DIR,
) -> list[TestCase]:
    """Discover all test cases, optionally filtered by task type and level.

    Scans testdata/cases/ for YAML files and parses them.

    Args:
        task_type: If set, only return cases matching this task type.
            Matches against the subdirectory name AND the task_type field.
        level: If set, only return cases that include this level
            (e.g., "L4" matches both "L4" and "L4+L5").
        root: Root directory to search. Defaults to testdata/cases/.

    Returns:
        Sorted list of TestCase objects (sorted by case_id).
    """
    ...


def golden_path_for_case(case: TestCase) -> Path:
    """Return the expected golden .npz path for a test case.

    Maps testdata/cases/<task>/<name>.yaml → testdata/golden/<task>/<name>.npz
    """
    ...


def has_golden(case: TestCase) -> bool:
    """Check whether the golden .npz file exists for a test case."""
    ...


def load_tolerances(
    level: str = "L4",
    dtype: str = "float32",
    path: Path = DEFAULT_TOLERANCES_PATH,
) -> Tolerances:
    """Load tolerance thresholds for a given level and dtype.

    Args:
        level: "L3", "L4", or "L5".
        dtype: "float32", "float16", "bfloat16", or "int4".
        path: Path to the YAML tolerances file.

    Returns:
        Tolerances dataclass with resolved thresholds.
    """
    ...
```

### Integration with Existing `compare_golden()`

The `golden.py` module does NOT replace `parity.py`. It feeds data INTO `compare_golden()`:

```python
# Usage pattern in tests:
from mobius._testing.golden import load_test_case, load_golden_ref, golden_path_for_case
from mobius._testing.parity import compare_golden, ParityResult

case = load_test_case(Path("testdata/cases/causal-lm/qwen2_5-0_5b.yaml"))
golden = load_golden_ref(golden_path_for_case(case))

# Run ONNX inference (produces onnx_logits)
onnx_logits = run_onnx_model(case)

# L4 check
report = compare_golden(
    onnx_logits=onnx_logits,
    golden_top1_id=golden.top1_id,
    golden_top2_id=golden.top2_id,
    golden_top10_ids=golden.top10_ids,
    dtype=case.dtype,
)
assert report.result != ParityResult.FAIL, report.message
```

---

## 6. `generate_golden.py` CLI

File: `scripts/generate_golden.py`

### Usage

```bash
# Generate golden files for ALL test cases
python scripts/generate_golden.py

# Generate for a specific task type
python scripts/generate_golden.py --task-type causal-lm

# Generate for a single test case
python scripts/generate_golden.py --case testdata/cases/causal-lm/qwen2_5-0_5b.yaml

# Regenerate all (overwrite existing)
python scripts/generate_golden.py --force

# Use GPU for large models
python scripts/generate_golden.py --device cuda

# Dry run (show what would be generated)
python scripts/generate_golden.py --dry-run

# Only generate L5 data (skip L4-only cases)
python scripts/generate_golden.py --level L5
```

### CLI Arguments

```python
"""Generate golden reference files for L4/L5 testing.

Reads YAML test case definitions from testdata/cases/ and generates
.npz golden reference files in testdata/golden/ by running HuggingFace
inference.

Requires: pip install transformers torch accelerate
GPU recommended for models > 1B parameters.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate golden reference files for L4/L5 testing"
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help="Path to a single YAML test case. If omitted, processes all.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=[
            "causal-lm", "encoder", "seq2seq",
            "vision-language", "vision", "audio",
        ],
        help="Only generate for this task type.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["L4", "L5"],
        help="Only generate cases that include this level.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running inference.",
    )
    return parser.parse_args()
```

### Core Generation Logic

```python
def generate_golden_for_case(
    case: TestCase,
    device: str = "cpu",
) -> None:
    """Generate golden reference data for a single test case.

    Dispatches to task-specific generation functions based on case.task_type.
    """
    npz_path = golden_path_for_case(case)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if case.task_type == "text-generation":
        _generate_causal_lm(case, npz_path, device)
    elif case.task_type == "feature-extraction":
        _generate_encoder(case, npz_path, device)
    elif case.task_type == "seq2seq":
        _generate_seq2seq(case, npz_path, device)
    elif case.task_type in ("qwen-vl", "vision-language", ...):
        _generate_vision_language(case, npz_path, device)
    elif case.task_type == "speech-to-text":
        _generate_speech_to_text(case, npz_path, device)
    elif case.task_type == "image-classification":
        _generate_vision(case, npz_path, device)
    else:
        print(f"  SKIP: unsupported task_type={case.task_type}")
        return

    print(f"  SAVED: {npz_path}")


def _generate_causal_lm(
    case: TestCase,
    npz_path: Path,
    device: str,
) -> None:
    """Generate golden data for a causal-lm model.

    Uses torch_reference.load_torch_model() and torch_forward() from
    the existing _testing infrastructure.
    """
    from mobius._testing.torch_reference import (
        load_torch_model,
        torch_forward,
    )

    model, tokenizer = load_torch_model(
        case.model_id, device=device,
    )

    # Tokenize
    encoded = tokenizer(
        case.prompts[0],
        return_tensors="np",
        padding=False,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len).reshape(1, -1)

    # L4: single forward pass
    logits, _ = torch_forward(
        model, input_ids, attention_mask, position_ids,
    )
    last_logits = logits[0, -1, :]  # (vocab_size,)
    sorted_indices = np.argsort(last_logits)[::-1]
    top10_ids = sorted_indices[:10].tolist()
    top10_logits = last_logits[sorted_indices[:10]].tolist()

    golden_data = {
        "top1_id": np.array([top10_ids[0]], dtype=np.int64),
        "top2_id": np.array([top10_ids[1]], dtype=np.int64),
        "top10_ids": np.array(top10_ids, dtype=np.int64),
        "top10_logits": np.array(top10_logits, dtype=np.float64),
        "logits_summary": np.array([
            float(np.max(last_logits)),
            float(np.min(last_logits)),
            float(np.mean(last_logits)),
            float(np.std(last_logits)),
        ], dtype=np.float64),
        "input_ids": input_ids.astype(np.int64),
    }

    # L5: greedy generation (if requested)
    if "L5" in case.level:
        import torch
        with torch.no_grad():
            gen_output = model.generate(
                torch.from_numpy(input_ids).to(device),
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                do_sample=False,
            )
        generated_ids = gen_output[0, seq_len:].cpu().numpy()
        golden_data["generated_ids"] = generated_ids.astype(np.int64)

    np.savez_compressed(npz_path, **golden_data)
```

### Handling Multi-Model Tasks (VL)

```python
def _generate_vision_language(
    case: TestCase,
    npz_path: Path,
    device: str,
) -> None:
    """Generate golden data for vision-language models.

    Loads the full HF multimodal model, runs the processor to
    prepare inputs (text + image), runs a single forward pass
    for L4, and optionally generates for L5.
    """
    from mobius._testing.torch_reference import (
        load_torch_multimodal_model,
    )

    model, tokenizer, processor = load_torch_multimodal_model(
        case.model_id, device=device,
    )

    # Prepare multimodal inputs
    from PIL import Image
    images = [
        Image.open(Path("testdata") / img_path)
        for img_path in case.images
    ]

    processed = processor(
        text=case.prompts[0],
        images=images if images else None,
        return_tensors="pt",
    ).to(device)

    # Forward pass for L4
    import torch
    with torch.no_grad():
        outputs = model(**processed)

    logits = outputs.logits[0, -1, :].cpu().numpy()
    sorted_indices = np.argsort(logits)[::-1]

    golden_data = {
        "decoder.top1_id": np.array([sorted_indices[0]], dtype=np.int64),
        "decoder.top2_id": np.array([sorted_indices[1]], dtype=np.int64),
        "decoder.top10_ids": np.array(sorted_indices[:10], dtype=np.int64),
        "decoder.top10_logits": np.array(
            logits[sorted_indices[:10]], dtype=np.float64
        ),
        "decoder.logits_summary": np.array([
            float(np.max(logits)),
            float(np.min(logits)),
            float(np.mean(logits)),
            float(np.std(logits)),
        ], dtype=np.float64),
        "decoder.input_ids": processed["input_ids"].cpu().numpy().astype(np.int64),
    }

    # L5: generation
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                **processed,
                max_new_tokens=case.generation_params.get("max_new_tokens", 30),
                do_sample=False,
            )
        input_len = processed["input_ids"].shape[1]
        generated_ids = gen[0, input_len:].cpu().numpy()
        golden_data["generated_ids"] = generated_ids.astype(np.int64)

    np.savez_compressed(npz_path, **golden_data)
```

### GPU Handling

The CLI uses `--device` for this. Key considerations:
- **CPU is the default** for CI reproducibility. Golden files generated on CPU are the canonical reference.
- **GPU is opt-in** for large models (>1B params) that are too slow on CPU.
- **GPU golden files should be regenerated on CPU** before committing, or the YAML should note `device: gpu` so CI knows to expect potential differences.
- **Recommendation:** Start with small models (0.5B-1B) that run on CPU. Add GPU models later when CI has GPU runners.

### Handling `trust_remote_code`

```python
if case.trust_remote_code:
    model, tokenizer = load_torch_model(
        case.model_id,
        device=device,
        trust_remote_code=True,  # Pass through to transformers
    )
```

The `trust_remote_code` flag in the YAML propagates to the HF loader. This is only needed for golden file *generation* (offline, developer machine). CI tests only load the `.npz` file — they never run HF inference.

---

## 7. `e2e_golden_test.py` Structure

File: `tests/e2e_golden_test.py`

### Parametrization Strategy

```python
"""L4 (Checkpoint Verified) and L5 (Generation E2E) golden tests.

Data-driven: each YAML file in testdata/cases/ is a test case.
Adding coverage = adding a YAML + .npz file. No code changes needed.

Run:
    pytest tests/e2e_golden_test.py -v                          # all
    pytest tests/e2e_golden_test.py -k "causal-lm"             # by task
    pytest tests/e2e_golden_test.py -k "qwen2_5-0_5b"          # by model
    pytest tests/e2e_golden_test.py -k "L4"                    # by level
    pytest tests/e2e_golden_test.py -m golden_fast              # small models only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mobius._builder import build, build_from_module
from mobius._registry import registry
from mobius._testing.golden import (
    TestCase,
    discover_test_cases,
    golden_path_for_case,
    has_golden,
    load_golden_ref,
    load_tolerances,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.parity import compare_golden, ParityResult


# --- Test case discovery (runs at collection time) ---

def _discover_and_parametrize() -> list[pytest.param]:
    """Discover all test cases and create pytest params.

    Missing golden files → skip (not fail). This allows adding YAML
    test cases before generating golden data.
    """
    cases = discover_test_cases()
    params = []
    for case in cases:
        marks = []
        # Mark by task type for -k filtering
        marks.append(pytest.mark.golden)

        if case.skip_reason:
            marks.append(pytest.mark.skip(reason=case.skip_reason))
        elif not has_golden(case):
            marks.append(pytest.mark.skip(
                reason=f"Golden file missing: {golden_path_for_case(case)}"
            ))

        params.append(pytest.param(
            case,
            id=f"{case.task_type}/{case.case_id}",
            marks=marks,
        ))
    return params


_GOLDEN_CASES = _discover_and_parametrize()


# --- L4 Tests (Checkpoint Verified) ---

class TestL4CheckpointVerified:
    """L4: Compare ONNX single-forward-pass output against golden argmax.

    Gate: argmax match (with near-tie AMBIGUOUS tolerance).
    """

    @pytest.mark.golden
    @pytest.mark.parametrize("case", _GOLDEN_CASES)
    def test_prefill_argmax_matches_golden(self, case: TestCase) -> None:
        if "L4" not in case.level:
            pytest.skip("Test case does not include L4")

        golden = load_golden_ref(golden_path_for_case(case))
        tolerances = load_tolerances("L4", case.dtype)

        # Build ONNX model from HF config
        pkg = build(case.model_id, trust_remote_code=case.trust_remote_code)

        # Prepare inputs from golden data (reuse tokenized input)
        session = OnnxModelSession(pkg)
        feeds = _prepare_feeds_for_task(case, golden, session)
        outputs = session.run(feeds)

        # Extract logits
        logits = _extract_logits(outputs, case.task_type)

        # Compare
        report = compare_golden(
            onnx_logits=logits,
            golden_top1_id=golden.top1_id,
            golden_top2_id=golden.top2_id,
            golden_top10_ids=golden.top10_ids,
            dtype=case.dtype,
        )

        # Warn on low Jaccard (not a failure)
        if report.top10_jaccard < tolerances.top10_jaccard_warn:
            import warnings
            warnings.warn(
                f"Low top-10 Jaccard: {report.top10_jaccard:.2f} "
                f"< {tolerances.top10_jaccard_warn}"
            )

        assert report.result != ParityResult.FAIL, report.message


# --- L5 Tests (Generation E2E) ---

class TestL5GenerationE2E:
    """L5: Compare ONNX greedy generation output against golden token sequence.

    Gate: exact token match (with per-dtype partial credit).
    """

    @pytest.mark.golden
    @pytest.mark.parametrize("case", _GOLDEN_CASES)
    def test_generation_matches_golden(self, case: TestCase) -> None:
        if "L5" not in case.level:
            pytest.skip("Test case does not include L5")

        golden = load_golden_ref(golden_path_for_case(case))
        tolerances = load_tolerances("L5", case.dtype)

        assert golden.generated_ids is not None, (
            f"Golden file for {case.case_id} has level=L5 but no generated_ids"
        )

        # Build and run ONNX generation
        pkg = build(case.model_id, trust_remote_code=case.trust_remote_code)
        generated_ids = _run_onnx_generation(
            pkg, case, golden,
            max_new_tokens=case.generation_params.get("max_new_tokens", 20),
        )

        # Compare token sequences
        match_ratio = _token_match_ratio(generated_ids, golden.generated_ids)

        assert match_ratio >= tolerances.min_token_match_ratio, (
            f"L5 FAIL: token match ratio {match_ratio:.2f} "
            f"< {tolerances.min_token_match_ratio:.2f}\n"
            f"  Expected: {golden.generated_ids.tolist()}\n"
            f"  Got:      {generated_ids.tolist()}"
        )


# --- Helpers ---

def _prepare_feeds_for_task(
    case: TestCase,
    golden: GoldenRef,
    session: OnnxModelSession,
) -> dict[str, np.ndarray]:
    """Prepare ONNX session input feeds based on task type.

    Uses the tokenized input_ids from the golden file to ensure
    reproducibility (same tokens as HF reference).
    """
    input_ids = golden.input_ids
    seq_len = input_ids.shape[1]
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Add KV cache (zeros) for decoder models
    for name in session.input_names:
        if name not in feeds:
            # Infer shape from the model's expected inputs
            feeds[name] = np.zeros(
                _infer_cache_shape(name, session), dtype=np.float32
            )

    return feeds


def _extract_logits(
    outputs: dict[str, np.ndarray],
    task_type: str,
) -> np.ndarray:
    """Extract the relevant output tensor based on task type."""
    if task_type in ("text-generation", "seq2seq"):
        return outputs["logits"]
    elif task_type == "feature-extraction":
        return outputs.get("last_hidden_state", outputs.get("logits"))
    else:
        return outputs["logits"]


def _token_match_ratio(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    """Compute the ratio of matching tokens between two sequences."""
    min_len = min(len(actual), len(expected))
    if min_len == 0:
        return 0.0
    matches = sum(
        1 for a, e in zip(actual[:min_len], expected[:min_len]) if a == e
    )
    return matches / len(expected)


def _run_onnx_generation(
    pkg,
    case: TestCase,
    golden: GoldenRef,
    max_new_tokens: int,
) -> np.ndarray:
    """Run greedy generation using ONNX model with manual decode loop.

    This mimics the ORT GenAI generation loop:
    1. Prefill: forward(input_ids) → logits + KV cache
    2. Decode: for each step, argmax → next token, forward(next_token)
    """
    session = OnnxModelSession(pkg)

    # Prefill
    feeds = _prepare_feeds_for_task(case, golden, session)
    outputs = session.run(feeds)
    logits = outputs["logits"]

    generated = []
    for step in range(max_new_tokens):
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

        # Prepare decode-step feeds (single token + KV cache)
        feeds = _prepare_decode_feeds(
            next_token, step, outputs, session, case
        )
        outputs = session.run(feeds)
        logits = outputs["logits"]

    return np.array(generated, dtype=np.int64)
```

### Missing Golden Files: Skip, Not Fail

**Critical design choice:** When a test case YAML exists but its golden `.npz` does not, the test is **skipped** (not failed). This enables the workflow:

1. Developer adds `testdata/cases/causal-lm/new-model.yaml`
2. Commits and pushes
3. CI runs — the new test case is discovered but skipped (golden missing)
4. Developer runs `python scripts/generate_golden.py --case testdata/cases/causal-lm/new-model.yaml`
5. Commits the `.npz` file
6. CI runs — the test now executes and passes

This separates "what we want to test" (YAML) from "what we have data for" (npz). YAML files can be committed eagerly.

### CI Configuration

```yaml
# .github/workflows/golden_tests.yaml
name: L4/L5 Golden Tests

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:
  pull_request:
    paths:
      - 'src/mobius/**'
      - 'testdata/cases/**'
      - 'testdata/golden/**'

jobs:
  golden-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[transformers,testing]"
      - run: python -m pytest tests/e2e_golden_test.py -v --tb=short -m golden

  # Separate job: regenerate golden files (manual trigger only)
  regenerate-golden:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest  # or GPU runner for large models
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[transformers,testing]" torch transformers
      - run: python scripts/generate_golden.py --force
      - uses: peter-evans/create-pull-request@v6
        with:
          title: "chore: regenerate golden files"
          branch: golden-refresh
```

---

## Appendix A: Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Golden format | `.npz` | Zero deps, fast, compact, named arrays |
| Full-vocab logits | NOT stored | Only ~100 ints per case needed; full logits are 128KB+ and regenerable |
| Directory grouping | By task type | Matches test runner dispatch logic; natural for `discover_test_cases()` |
| Missing golden | Skip | Enables YAML-first workflow; separates intent from data |
| L5 tolerance | Token match ratio | Hard exact-match for f32; partial credit for quantized |
| Multi-model golden | Dotted keys in npz | `decoder.top1_id`, `vision.output_norm` — flat namespace, no nesting |
| Generation params | Always greedy | Deterministic. Sampling is non-reproducible across platforms. |
| Filename convention | Model ID → slug | Predictable, grep-friendly, no spaces or special chars |

## Appendix B: Migration Path from Existing `integration_test.py`

The existing `integration_test.py` runs HF inference live. The golden file infrastructure replaces this for daily CI:

| Before (integration_test.py) | After (e2e_golden_test.py) |
|------|------|
| Downloads HF model on every run | Uses pre-computed .npz golden data |
| Requires GPU for large models | CPU-only (ONNX inference) |
| ~5 min for 12 text models | ~30 sec for same models (no HF download) |
| Tests parametrized in Python code | Tests parametrized from YAML files |
| Adding coverage = code change | Adding coverage = YAML + npz file |

**`integration_test.py` is NOT deleted.** It remains as the "source of truth" for generating golden data and as a development-time tool. The golden tests are its pre-computed counterpart.

## Appendix C: File Size Estimates

| Component | Per case | 100 cases |
|-----------|----------|-----------|
| YAML test case | ~300 bytes | ~30 KB |
| Golden .npz (L4) | ~2 KB | ~200 KB |
| Golden .npz (L4+L5) | ~3 KB | ~300 KB |
| Tolerance config | ~500 bytes | ~500 bytes |
| **Total** | | **~530 KB** |

All comfortably within git repo limits. No Git LFS needed.
