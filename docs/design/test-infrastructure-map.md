# ONNX GenAI Models - Test Infrastructure Comprehensive Map

Last Updated: 2025-03-09

---

## EXECUTIVE SUMMARY

| Aspect | Count | Location |
|--------|-------|----------|
| **Test Files** | 37 | `/tests/` (18) + `src/**/*_test.py` (19+) |
| **Test Classes** | 248 | 33 in `/tests/`, 204 in `src/` |
| **Test Methods** | 237 | in `/tests/` (79) + `src/` (1003) |
| **Test Fixtures** | 1 | `conftest.py` (deterministic_seed) |
| **Pytest Markers** | 7 | `integration`, `integration_fast`, `integration_slow`, `arch_validation`, `golden`, `generation`, `benchmark` |
| **CI Workflows** | 7 | main, gpu_tests, nightly_l2, golden_regen, arch_diff, benchmark, pages |
| **Test Dependencies** | 10 | pytest, pytest-xdist, pytest-timeout, onnxruntime-easy, transformers, safetensors, gguf, Pillow, librosa, onnxruntime-genai |

---

## 1. TEST DIRECTORIES & FILES

### 1.1 `/tests/` Directory Structure

**Main Test Files (18 total):**

| File | Classes | Methods | Purpose | Markers |
|------|---------|---------|---------|---------|
| `build_graph_test.py` | 33 | 100 | **L1 Unit Tests**: Build ONNX graphs for all supported architectures without weights or network access | `parametrize` |
| `arch_validation_test.py` | 1 | 4 | **L2 Architecture Validation**: Downloads config.json from HF for registered models, builds full-size ONNX graphs (no weights), validates structure | `arch_validation`, `parametrize` |
| `synthetic_parity_test.py` | 0 | 1 | **L3 Synthetic Parity**: Builds tiny random-weight models (PyTorch + ONNX), compares logits to detect op-level bugs | `parametrize` |
| `e2e_golden_test.py` | 2 | 2 | **L4/L5 Golden Comparison**: Data-driven tests from YAML files in `testdata/cases/`; L4 compares single-forward logits, L5 compares generation | `golden`, `generation`, `integration`, `parametrize` |
| `integration_test.py` | 15 | 34 | **Integration Tests**: Downloads real model weights, compares logits and greedy generation vs HF PyTorch reference | `integration`, `integration_fast`, `integration_slow`, `parametrize` |
| `weight_alignment_test.py` | 5 | 5 | **Weight Alignment Tests**: Verifies preprocess_weights() doesn't corrupt ONNX names; catches renaming bugs | `parametrize` |
| `onnx_checker_test.py` | 1 | 1 | **ONNX Checker Tests**: Runs CheckerPass on all built models; detects op errors, malformed protos (~30s) | `parametrize` |
| `cli_test.py` | 3 | 8 | **CLI Tests**: Invokes `main()` directly; no network access, all build tests use `--no-weights` | None |
| `seq2seq_integration_test.py` | 8 | 9 | **Seq2Seq Integration**: T5/BART encoder-decoder vs HF PyTorch | `integration`, `parametrize` |
| `vision_integration_test.py` | 2 | 2 | **Vision Integration**: ViT/CLIP with random weights vs PyTorch | `integration`, `integration_fast` |
| `multimodal_integration_test.py` | 1 | 2 | **Multimodal Integration**: Vision-language models (Gemma3, etc.) | `integration`, `integration_slow` |
| `moe_integration_test.py` | 2 | 3 | **MoE Integration**: Deprecated (superseded by `integration_test.py`); MoE models prefill/decode | `integration`, `integration_fast` |
| `whisper_integration_test.py` | 3 | 4 | **Whisper Integration**: Audio encoder-decoder vs HF | `integration`, `integration_fast` |
| `ort_genai_test.py` | 2 | 3 | **ONNX Runtime GenAI**: End-to-end with onnxruntime-genai inference | `integration`, `integration_slow`, `parametrize` |
| `mamba2_integration_test.py` | 1 | 2 | **Mamba2 Integration**: SSM state carry, single-token decode | `integration` |
| `phi4mm_integration_test.py` | 1 | 4 | **Phi4-MM Integration**: 4-model pipeline (vision, speech, embedding, decoder) with text+image+audio | `integration`, `integration_slow` |
| `quantization_integration_test.py` | 2 | 6 | **Quantization Integration**: Tiny quantized Llama, synthetic weights, MatMulNBits execution | `integration` |
| `benchmark_build.py` | 1 | 1 | **Build Performance Benchmark**: Measures `task.build()` time and memory for key architectures | `benchmark`, `parametrize` |

**Support Files:**
- `conftest.py` (141 lines) — Shared fixtures, marker registration, CLI flags, test filtering
- `_test_configs.py` (38KB) — Config data for parametrized tests

### 1.2 `src/` Test Files (54 total)

**Test Coverage by Module:**

| Module | Classes | Methods | Key Test Files |
|--------|---------|---------|-----------------|
| **Core Config** | 27 | 189 | `_configs_test.py` (62M), `_config_resolver_test.py` (40M), `_registry_test.py` (27M) |
| **Building & Export** | 22 | 113 | `_exporter_test.py` (32M), `_model_package_test.py` (29M), `_diffusers_builder_test.py` (21M) |
| **Components** | 67 | 379 | 23 test files (rotary_embedding 28M, quantized_linear 22M, audio 16M, attention 17M, vision 16M, whisper 20M, etc.) |
| **Weight Handling** | 17 | 114 | `_weight_utils_test.py` (58M), `_weight_loading_test.py` (27M), `_graph_diff_test.py` (24M) |
| **GGUF Integration** | 19 | 91 | `_builder_test.py`, `_reader_test.py` (26M), `_repacker_test.py` (30M), `_tensor_mapping_test.py` (18M) |
| **Testing Utilities** | 9 | 55 | `golden_test.py` (28M), `parity_test.py` (11M), `code_paths_test.py` (16M) |
| **Tasks Framework** | 15 | 81 | `_task_test.py` (15 classes, 81 methods — core model building logic) |
| **Other** | 27 | 61 | ORT GenAI auto_export_test (integration marker), rewrite rules (bias/gelu/layer_norm fusion, etc.) |

**Total: 204 test classes, 1003 test methods in src/**

---

## 2. CONFTEST.PY FILES

### `/tests/conftest.py` (141 lines)

**Marker Registrations (via `pytest_configure`):**
```python
markers = [
    "golden: L4/L5 golden reference comparison tests",
    "generation: L5 end-to-end generation tests",
    "integration: tests that require real model weights (network)",
    "integration_fast: fast integration tests with small models",
    "arch_validation: L2 architecture validation tests",
]
```

**CLI Flags (via `pytest_addoption`):**
- `--fast`: Skip non-representative model configs (uses FAST_*_CONFIGS)
- `--models <comma-separated-list>`: Filter by model_type parameter

**Collection Hook (`pytest_collection_modifyitems`):**
- Filters parametrized tests based on `--fast` and `--models` flags
- Respects `model_type` parameter in test callspecs
- Always runs non-parametrized tests

**Fixtures:**
```python
@pytest.fixture()
def deterministic_seed():
    """Set numpy and python random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    return seed
```

**Note:** No `conftest.py` files found in `src/` directory (tests are colocated with modules).

---

## 3. TEST UTILITIES

### `/src/mobius/_testing/` (12 files)

**Exported Classes & Functions:**

| Module | Key Classes/Functions |
|--------|----------------------|
| `__init__.py` | Public API: `GoldenRef`, `TestCase`, `Tolerances`, `discover_test_cases()`, `load_golden_ref()`, `save_golden_ref()`, `load_test_case()`, `load_tolerances()`, `compare_golden()`, `compare_synthetic()`, `detect_code_paths()`, `CODE_PATH_INDICATORS`, etc. |
| `golden.py` | `GoldenRef` — stores L4/L5 reference outputs; `TestCase` — YAML test case (model_id, task, seed); `Tolerances` — L4/L5 comparison thresholds; `discover_test_cases()`, `load_golden_ref()`, `save_golden_ref()`, `load_test_case()`, `load_tolerances()`, `golden_path_for_case()`, `has_golden()` |
| `parity.py` | `ParityReport`, `ParityResult` — numerical comparison results; `compare_golden()` — L4 logit comparison; `compare_synthetic()` — L3 synthetic parity check |
| `code_paths.py` | `CodePathIndicator` — config→code-path mapping; `detect_code_paths()`, `detect_code_paths_from_config()`, `get_all_code_path_labels()` — implicit code path detection from config values |
| `generation.py` | `OnnxGenerator` — greedy autoregressive text generation with KV cache; `generate()` method |
| `ort_inference.py` | `OnnxModelSession` — ONNX Runtime inference wrapper |
| `torch_reference.py` | `TorchReference` — PyTorch reference model wrapper for parity testing |
| `comparison.py` | `compare_onnx_and_torch()` — numerical comparison utilities |
| `code_paths_test.py` | Tests for code-path detection (2 classes, 16 methods) |
| `golden_test.py` | Tests for golden-file loading/saving (5 classes, 28 methods) |
| `parity_test.py` | Tests for parity utilities (2 classes, 11 methods) |

**Test Infrastructure Design:**
- **Golden files** stored in `testdata/golden/` (YAML + JSON)
- **Test cases** in `testdata/cases/` (YAML definitions)
- **Tolerances** in `testdata/default_tolerances.yaml` (L4/L5 thresholds)
- **Data-driven approach**: Adding test coverage = adding a YAML file (no code changes needed)

---

## 4. CI CONFIGURATION & WORKFLOWS

### `.github/workflows/` (7 files)

| File | Trigger | Level | Test Command | Purpose |
|------|---------|-------|--------------|---------|
| **main.yml** | PR + push to main | L1–L3, selective L4 | `pytest tests/build_graph_test.py -n auto` (L1); `pytest tests/synthetic_parity_test.py -n auto` (L3); `pytest -m 'not integration'` (coverage); selective L4 + L5 | Fast PR feedback; runs on every push |
| **gpu_tests.yml** | push to main + weekly (Mon 4am UTC) | L4–L5 | `pytest tests/e2e_golden_test.py -m golden` (L4); `pytest tests/e2e_golden_test.py -m generation` (L5) | GPU-only golden/generation tests (requires checkpoint verification) |
| **nightly_l2.yml** | nightly (3am UTC) + l2-* tags | L2 | `pytest tests/arch_validation_test.py -m arch_validation` | Architecture validation with real HF configs |
| **golden_regen.yml** | weekly (Sun 2am UTC) + manual | — | Regenerates golden files | Updates L4/L5 reference data (before nightly_l2) |
| **arch_diff.yml** | Manual dispatch | — | Detects architecture differences | Custom diff detection (details TBD) |
| **benchmark.yml** | Manual dispatch + scheduled | — | Benchmark reporting | Performance tracking |
| **pages.yml** | Push to main | — | Docs build | GitHub Pages deployment |

**CI Test Levels:**
```
L1 (Smoke)           → build_graph_test.py            (fast, no weights, no network)
L2 (Architecture)    → arch_validation_test.py        (real HF configs, no weights, nightly)
L3 (Synthetic)       → synthetic_parity_test.py       (tiny random weights, detects op bugs)
L4 (Golden Logits)   → e2e_golden_test.py -m golden   (checkpoint-verified single-forward)
L5 (Generation)      → e2e_golden_test.py -m generation (checkpoint-verified generation)
```

**Caching Strategy:**
- `pip-smoke-*`, `pip-parity-*`, `pip-gpu-*`, `pip-nightly-l2-*` — pip package cache
- `hf-gpu-*`, `hf-l2-configs-*` — HuggingFace model/config cache

---

## 5. PYTEST CONFIGURATION

### `pyproject.toml` [tool.pytest.ini_options]

```python
[tool.pytest.ini_options]
addopts = "--tb=short --color=yes"
testpaths = ["src", "tests"]
python_files = ["*_test.py"]
markers = [
    "integration: tests requiring model downloads and GPU/CPU inference",
    "integration_fast: fast integration tests (text, encoder, audio)",
    "integration_slow: slow integration tests (vision-language, multimodal)",
    "benchmark: graph construction performance benchmarks",
    "arch_validation: L2 architecture validation (real HF configs, no weights)",
    "golden: L4 checkpoint-verified golden comparison tests",
    "generation: L5 end-to-end golden generation tests",
]
```

**pytest Options:**
- `--tb=short` — Short traceback format
- `--color=yes` — Colorized output
- `testpaths` — Test discovery paths: `src/**/*_test.py` + `tests/**/*_test.py`
- `python_files` — Only files matching `*_test.py`

---

## 6. TEST DEPENDENCIES

### `pyproject.toml` [project.optional-dependencies]

```python
testing = [
    "onnxruntime-easy",           # CPU inference
    "torch",                       # PyTorch reference
    "transformers>=5.0",          # HuggingFace models
    "safetensors",                # Weight loading
    "onnxruntime-genai",          # GenAI runtime
    "pytest-xdist",               # Parallel test execution (-n auto)
    "pytest-timeout",             # Timeout management
    "gguf>=0.10.0",               # GGUF format
    "Pillow",                      # Image processing
    "librosa",                     # Audio processing
]
```

### `requirements/ci/requirements.txt`

```
pytest
pytest-cov
pytest-xdist
coverage[toml]
gguf>=0.10.0
```

---

## 7. PYTEST MARKERS USAGE

### Marker Hierarchy

| Marker | Level | When to Use | Deselection |
|--------|-------|-------------|-------------|
| (none) | L0–L1 | Unit tests, fast integration | Always run |
| `@pytest.mark.parametrize` | All | Model/config parametrization | Run as needed |
| `@pytest.mark.arch_validation` | L2 | Architecture validation | `-m "not arch_validation"` |
| `@pytest.mark.golden` | L4 | Golden logit comparison | `-m "not golden"` |
| `@pytest.mark.generation` | L5 | Golden generation E2E | `-m "not generation"` |
| `@pytest.mark.integration` | L3–L5 | Real weights required | `-m "not integration"` |
| `@pytest.mark.integration_fast` | L3 | Small models (text, encoder, audio) | Subset of integration |
| `@pytest.mark.integration_slow` | L3 | Large models (multimodal, VL) | Subset of integration |
| `@pytest.mark.benchmark` | — | Performance benchmarking | `-m "not benchmark"` |
| `@pytest.mark.skip` | — | Conditional skips | N/A |

### Marker Usage Counts (in `/tests/*.py`)

```
@pytest.mark.integration        56 usages
@pytest.mark.integration_fast   36 usages
@pytest.mark.integration_slow    9 usages
@pytest.mark.golden              1 usage (e2e_golden_test.py)
@pytest.mark.generation          1 usage (e2e_golden_test.py)
@pytest.mark.arch_validation     1 usage (arch_validation_test.py)
@pytest.mark.benchmark           1 usage (benchmark_build.py)
@pytest.mark.parametrize        Multiple (config/model parametrization)
```

---

## 8. FIXTURES

### Fixture Definitions

**`/tests/conftest.py`:**
```python
@pytest.fixture()
def deterministic_seed():
    """Set numpy and python random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    return seed
```

**`src/mobius/integrations/gguf/_builder_test.py`:**
```python
@pytest.fixture
def [fixture_name]  # Implementation details TBD
```

**`src/mobius/integrations/gguf/_reader_test.py`:**
```python
@pytest.fixture
def [fixture_name_1]  # Implementation details TBD

@pytest.fixture
def [fixture_name_2]  # Implementation details TBD
```

**Total Fixture Count:** 4 (1 in tests/conftest.py, 3 in src/gguf/)

---

## 9. TEST ORGANIZATION SUMMARY

### Test Distribution by Type

| Type | Count | Location | Purpose |
|------|-------|----------|---------|
| **Unit (L0/L1)** | ~100 methods | `src/**/*_test.py`, `tests/build_graph_test.py` | Fast, no network |
| **Component** | ~379 methods | `src/components/*_test.py` | Individual operator validation |
| **Integration (L3)** | ~52 methods | `tests/*_integration_test.py`, `tests/synthetic_parity_test.py` | Real weights, network required |
| **Golden (L4/L5)** | ~2–10 methods | `tests/e2e_golden_test.py`, parametrized via YAML | Data-driven, checkpoint-verified |
| **Architecture (L2)** | 4 methods | `tests/arch_validation_test.py` | Real HF configs, no weights |
| **Infrastructure** | ~55 methods | `src/*_testing*/*_test.py` | Golden/parity utils testing |
| **Special** | ~17 methods | CLI, benchmarks, weight alignment | Auxiliary validation |

### Execution Strategy

1. **PR CI (main.yml):** L1 (smoke) + L3 (synthetic parity) + selective L4 (fast golden)
2. **Weekly GPU (gpu_tests.yml):** L4 (golden) + L5 (generation) on GPU
3. **Nightly (nightly_l2.yml):** L2 (architecture validation)
4. **Weekly Regen (golden_regen.yml):** Update L4/L5 golden data

---

## 10. KEY TESTING PATTERNS

### Pattern 1: Parametrized Model Testing
```python
@pytest.mark.parametrize("model_type,config_overrides", _MODEL_PARAMS)
def test_build_model(model_type, config_overrides):
    # Runs for each model_type in _MODEL_PARAMS
    # Filter with --fast flag or --models option
```

### Pattern 2: Integration Testing with Real Weights
```python
@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _TEXT_MODELS)
def test_export_and_run(model_id, trust_remote_code):
    # Downloads weights, compares logits
```

### Pattern 3: Golden-File-Based Testing (Data-Driven)
```python
@pytest.mark.golden
@pytest.mark.parametrize("case", _L4_CASES)
def test_golden_logits(case):
    # case = TestCase(model_id, task, seed, ...)
    # Loads from testdata/golden/*.json
```

### Pattern 4: Synthetic Parity Testing
```python
def test_parity_causal_lm():
    # Build PyTorch + ONNX with same seed
    # Compare logits for op-level bugs
```

---

## 11. TEST DATA LOCATIONS

| Path | Contents | Purpose |
|------|----------|---------|
| `testdata/cases/` | YAML test case definitions | L4/L5 test data (model_id, task, seed) |
| `testdata/golden/` | JSON golden reference outputs | L4/L5 expected results |
| `testdata/default_tolerances.yaml` | L4/L5 comparison thresholds | Numerical tolerance config |
| `tests/perf_baseline.json` | Baseline performance metrics | Benchmark regression guard |
| `tests/_test_configs.py` | Parametrization config data | FAST_*_CONFIGS, model lists, overrides |

---

## 12. QUICK REFERENCE COMMANDS

```bash
# L1 Smoke (all architectures)
pytest tests/build_graph_test.py -v

# L1 Fast (representative only)
pytest tests/build_graph_test.py -v --fast

# L2 Architecture Validation
pytest tests/arch_validation_test.py -m arch_validation -v

# L3 Synthetic Parity
pytest tests/synthetic_parity_test.py -v --tb=short -n 0

# L4 Golden Logits
pytest tests/e2e_golden_test.py -m golden -v

# L5 Generation E2E
pytest tests/e2e_golden_test.py -m generation -v

# Integration (small models only)
pytest tests/integration_test.py -m integration_fast -v -k "smollm or albert"

# All non-integration tests with coverage
pytest -m 'not integration' --cov=src --cov-report=html

# Filter by model type
pytest tests/build_graph_test.py -v --models "qwen2,llama"

# Run specific test
pytest tests/build_graph_test.py::TestQwen2::test_qwen2 -v
```

---

## 13. DESIGN NOTES

### Data-Driven Golden Testing
- **No hardcoded test code** for L4/L5 — each YAML file = one test case
- **Tolerances** centralized in `testdata/default_tolerances.yaml`
- **Golden data** stored as JSON (single-forward logits, generation tokens)
- **Benefits:** Easy to add new models without code changes, consistent thresholds

### Parametrization Strategy
- **Config-based:** Model parameters derived from `_test_configs.py`
- **Model-based:** Real HF model IDs parametrized via `@pytest.mark.parametrize`
- **Filtering:** `--fast` skips non-representative configs; `--models` filters by type

### Test Levels & CI Integration
- **L1 (Smoke):** Always runs, fastest feedback
- **L2 (Architecture):** Nightly, validates real HF configs
- **L3 (Parity):** Catches op bugs early, before L4/L5
- **L4/L5 (Golden):** Weekly on GPU, checkpoint-verified
- **Pyramid approach:** Fast tests first, expensive tests later

### Fixture Philosophy
- Minimal fixtures (just `deterministic_seed`)
- Most setup inline or in helper functions (e.g., `make_config()`)
- Avoids fixture complexity; easier to understand test flow

---

