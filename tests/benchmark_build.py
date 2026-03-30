# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for ONNX graph construction times and memory usage.

Measures how long ``task.build()`` takes and how much memory it uses for
key model architectures using tiny configs (no weights, no network).
Run as a standalone script for a formatted table::

    python tests/benchmark_build.py

Or via pytest to exercise the regression guard::

    pytest tests/benchmark_build.py -v
"""

from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Shared tiny dimensions (same as _test_configs)
# ---------------------------------------------------------------------------
sys.path.insert(0, "tests")
from _test_configs import (
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
@dataclass
class _BenchEntry:
    """A model to benchmark."""

    model_type: str
    config_overrides: dict
    task_name: str
    build_fn: str  # "standard" or "custom"


def _base_config(config_cls=None, **overrides):
    from mobius._configs import ArchitectureConfig

    if config_cls is None:
        config_cls = overrides.pop("_config_cls", ArchitectureConfig)
    else:
        overrides.pop("_config_cls", None)
    defaults = dict(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_HEAD_DIM,
        num_hidden_layers=TINY_LAYERS,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
    )
    defaults.update(overrides)
    return config_cls(**defaults)


def _display_key(model_type: str, task_name: str) -> str:
    """Build a unique display key for a model entry.

    Returns just ``model_type`` for the default ``text-generation`` task
    (backward compatible), or ``model_type (task_name)`` otherwise.
    """
    if task_name == "text-generation":
        return model_type
    return f"{model_type} ({task_name})"


# Top 10 diverse models spanning causal-LM, encoder-only, and seq2seq tasks.
BENCHMARK_MODELS: list[_BenchEntry] = [
    _BenchEntry("llama", {}, "text-generation", "standard"),
    _BenchEntry("llama", {}, "static-cache", "standard"),
    _BenchEntry("qwen2", {}, "text-generation", "standard"),
    _BenchEntry("qwen2", {}, "static-cache", "standard"),
    _BenchEntry(
        "phi3",
        {"partial_rotary_factor": 0.5},
        "text-generation",
        "standard",
    ),
    _BenchEntry(
        "phi3",
        {"partial_rotary_factor": 0.5},
        "static-cache",
        "standard",
    ),
    _BenchEntry(
        "gemma2",
        {
            "_config_cls": "Gemma2Config",
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
        },
        "text-generation",
        "standard",
    ),
    _BenchEntry(
        "falcon",
        {"attn_qkv_bias": True},
        "text-generation",
        "standard",
    ),
    _BenchEntry(
        "gpt2",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        "text-generation",
        "standard",
    ),
    _BenchEntry(
        "bert",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        "feature-extraction",
        "standard",
    ),
    _BenchEntry(
        "t5",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        "seq2seq",
        "standard",
    ),
    _BenchEntry("whisper", {}, "speech-to-text", "custom"),
    _BenchEntry("mamba", {}, "ssm-text-generation", "custom"),
    _BenchEntry(
        "qwen3_5_text",
        {
            "partial_rotary_factor": 0.5,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        "hybrid-text-generation",
        "standard",
    ),
    _BenchEntry(
        "qwen3_5_moe",
        {
            "hidden_act": "silu",
            "layer_types": ["linear_attention", "full_attention"],
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        "hybrid-text-generation",
        "standard",
    ),
    _BenchEntry("qwen3_5_vl", {}, "hybrid-qwen-vl", "custom"),
]


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
@dataclass
class _BuildResult:
    """Result of a single build."""

    model_type: str
    elapsed_s: float
    num_nodes: int
    num_models: int
    peak_memory_mb: float
    model_size_bytes: int = 0


def _measure_model_size(pkg) -> int:
    """Compute total parameter size in bytes across all models.

    Counts all initializers: those with materialized values use
    ``nbytes`` directly; weight placeholders (const_value is None)
    compute expected size from shape and dtype.
    """
    import math

    import numpy as np

    total = 0
    for model in pkg.values():
        for init in model.graph.initializers.values():
            if init.const_value is not None:
                total += init.const_value.nbytes
            elif init.shape is not None and init.dtype is not None:
                num_elements = math.prod(d for d in init.shape if isinstance(d, int))
                itemsize = np.dtype(init.dtype.numpy()).itemsize
                total += num_elements * itemsize
    return total


def _build_standard(entry: _BenchEntry) -> _BuildResult:
    """Build a model using the standard registry + task.build() path."""
    from mobius._registry import registry
    from mobius.tasks import CausalLMTask, get_task

    overrides = dict(entry.config_overrides)
    # Resolve _config_cls string to actual class if present
    cls_name = overrides.pop("_config_cls", None)
    if cls_name is not None:
        import mobius._configs as _cfgs

        config_cls = getattr(_cfgs, cls_name)
        config = _base_config(config_cls=config_cls, **overrides)
    else:
        config = _base_config(**overrides)

    model_cls = registry.get(entry.model_type)
    module = model_cls(config)
    if entry.task_name == "static-cache":
        task = CausalLMTask(static_cache=True)
    else:
        task = get_task(entry.task_name)

    tracemalloc.start()
    t0 = time.perf_counter()
    pkg = task.build(module, config)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_nodes = 0
    for model in pkg.values():
        total_nodes += len(list(model.graph))

    model_size = _measure_model_size(pkg)

    return _BuildResult(
        model_type=entry.model_type,
        elapsed_s=elapsed,
        num_nodes=total_nodes,
        num_models=len(pkg),
        peak_memory_mb=peak / 1024 / 1024,
        model_size_bytes=model_size,
    )


def _build_whisper() -> _BuildResult:
    """Build Whisper using its dedicated config and task."""
    from mobius._builder import build_from_module
    from mobius._configs import WhisperConfig
    from mobius.models.whisper import (
        WhisperForConditionalGeneration,
    )
    from mobius.tasks import SpeechToTextTask

    config = WhisperConfig(
        vocab_size=512,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_HEADS,
        head_dim=TINY_HIDDEN // TINY_HEADS,
        hidden_act="gelu",
        pad_token_id=0,
        tie_word_embeddings=True,
        attn_qkv_bias=True,
        attn_o_bias=True,
        encoder_layers=TINY_LAYERS,
        encoder_attention_heads=TINY_HEADS,
        encoder_ffn_dim=TINY_INTERMEDIATE,
        num_mel_bins=16,
        max_source_positions=100,
        max_target_positions=50,
        scale_embedding=True,
    )
    module = WhisperForConditionalGeneration(config)
    task = SpeechToTextTask()

    tracemalloc.start()
    t0 = time.perf_counter()
    pkg = build_from_module(module, config, task=task)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_nodes = 0
    for model in pkg.values():
        total_nodes += len(list(model.graph))

    model_size = _measure_model_size(pkg)

    return _BuildResult(
        model_type="whisper",
        elapsed_s=elapsed,
        num_nodes=total_nodes,
        num_models=len(pkg),
        peak_memory_mb=peak / 1024 / 1024,
        model_size_bytes=model_size,
    )


def _build_mamba() -> _BuildResult:
    """Build Mamba SSM using its dedicated config and task."""
    from mobius._builder import build_from_module
    from mobius._configs import MambaConfig
    from mobius.models.mamba import MambaCausalLMModel
    from mobius.tasks import SSMCausalLMTask

    config = MambaConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_HIDDEN * 2,
        num_hidden_layers=TINY_LAYERS,
        state_size=8,
        conv_kernel=4,
        expand=2,
        time_step_rank=4,
        layer_norm_epsilon=1e-5,
        use_conv_bias=True,
        tie_word_embeddings=True,
    )
    module = MambaCausalLMModel(config)
    task = SSMCausalLMTask()

    tracemalloc.start()
    t0 = time.perf_counter()
    pkg = build_from_module(module, config, task=task)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_nodes = 0
    for model in pkg.values():
        total_nodes += len(list(model.graph))

    model_size = _measure_model_size(pkg)

    return _BuildResult(
        model_type="mamba",
        elapsed_s=elapsed,
        num_nodes=total_nodes,
        num_models=len(pkg),
        peak_memory_mb=peak / 1024 / 1024,
        model_size_bytes=model_size,
    )


def _build_qwen3_5_vl() -> _BuildResult:
    """Build Qwen3.5-VL using its VisionConfig and hybrid task."""
    from mobius._configs import VisionConfig
    from mobius._registry import registry
    from mobius.tasks import get_task

    config = _base_config(
        attn_qk_norm=True,
        partial_rotary_factor=0.5,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            in_channels=3,
            out_hidden_size=64,
            num_position_embeddings=16,
        ),
        temporal_patch_size=2,
        spatial_merge_size=2,
        deepstack_visual_indexes=[0],
        image_token_id=248056,
        mrope_section=[8, 12, 12],
    )
    model_cls = registry.get("qwen3_5_vl")
    module = model_cls(config)
    task = get_task("hybrid-qwen-vl")

    tracemalloc.start()
    t0 = time.perf_counter()
    pkg = task.build(module, config)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_nodes = 0
    for model in pkg.values():
        total_nodes += len(list(model.graph))

    model_size = _measure_model_size(pkg)

    return _BuildResult(
        model_type="qwen3_5_vl",
        elapsed_s=elapsed,
        num_nodes=total_nodes,
        num_models=len(pkg),
        peak_memory_mb=peak / 1024 / 1024,
        model_size_bytes=model_size,
    )


def _run_single(entry: _BenchEntry) -> _BuildResult:
    """Dispatch to the right build function."""
    if entry.model_type == "whisper":
        return _build_whisper()
    if entry.model_type == "mamba":
        return _build_mamba()
    if entry.model_type == "qwen3_5_vl":
        return _build_qwen3_5_vl()
    return _build_standard(entry)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmarks(
    models: list[_BenchEntry] | None = None,
    iterations: int = 3,
) -> list[tuple[str, float, float, int, int, float, int]]:
    """Run benchmarks and return results per model.

    Args:
        models: List of models to benchmark. Defaults to BENCHMARK_MODELS.
        iterations: Number of times to build each model.

    Returns:
        List of (display_key, mean_seconds, std_seconds, num_nodes,
        num_models, peak_memory_mb, model_size_bytes).
    """
    if models is None:
        models = BENCHMARK_MODELS

    results: list[tuple[str, float, float, int, int, float, int]] = []
    for entry in models:
        times: list[float] = []
        last_result: _BuildResult | None = None
        for _ in range(iterations):
            result = _run_single(entry)
            times.append(result.elapsed_s)
            last_result = result
        assert last_result is not None
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        results.append(
            (
                _display_key(entry.model_type, entry.task_name),
                mean,
                std,
                last_result.num_nodes,
                last_result.num_models,
                last_result.peak_memory_mb,
                last_result.model_size_bytes,
            )
        )
    return results


def print_table(
    results: list[tuple[str, float, float, int, int, float, int]],
) -> None:
    """Print a formatted benchmark table to stdout."""
    header = (
        f"{'Model':<40} {'Mean (s)':>10} {'Std (s)':>10}"
        f" {'Nodes':>8} {'Models':>8} {'Peak MB':>10}"
        f" {'Size (KB)':>10}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, mean, std, nodes, n_models, peak_mb, size_bytes in results:
        print(
            f"{name:<40} {mean:>10.4f} {std:>10.4f}"
            f" {nodes:>8} {n_models:>8} {peak_mb:>10.1f}"
            f" {size_bytes / 1024:>10.1f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Pytest regression guard
# ---------------------------------------------------------------------------
MAX_BUILD_TIME_SECONDS = 10.0
MAX_PEAK_MEMORY_MB = 200.0

_BENCH_PARAMS = [
    pytest.param(entry, id=_display_key(entry.model_type, entry.task_name))
    for entry in BENCHMARK_MODELS
]


@pytest.mark.benchmark
@pytest.mark.parametrize("entry", _BENCH_PARAMS)
class TestBuildTimeRegression:
    """Guard against graph construction time and memory regressions.

    Each model must build in under MAX_BUILD_TIME_SECONDS and use less
    than MAX_PEAK_MEMORY_MB with a tiny config.
    """

    def test_build_time_under_threshold(self, entry: _BenchEntry):
        result = _run_single(entry)
        assert result.elapsed_s < MAX_BUILD_TIME_SECONDS, (
            f"{entry.model_type} took {result.elapsed_s:.2f}s "
            f"(threshold: {MAX_BUILD_TIME_SECONDS}s)"
        )
        assert result.num_nodes > 0, f"{entry.model_type} produced 0 nodes"
        assert result.peak_memory_mb < MAX_PEAK_MEMORY_MB, (
            f"{entry.model_type} used {result.peak_memory_mb:.1f}MB "
            f"(threshold: {MAX_PEAK_MEMORY_MB}MB)"
        )
        assert result.model_size_bytes > 0, f"{entry.model_type} produced 0-byte model"


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def _get_git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_json(
    results: list[tuple[str, float, float, int, int, float, int]],
    path: str,
) -> None:
    """Write benchmark results as JSON for CI comparison."""
    import json

    data: dict = {
        "_metadata": {
            "commit": _get_git_commit(),
            "python": sys.version.split()[0],
        },
        "models": {},
    }
    for (
        name,
        _mean,
        _std,
        nodes,
        n_models,
        _peak_mb,
        size_bytes,
    ) in results:
        data["models"][name] = {
            "num_nodes": nodes,
            "num_models": n_models,
            "model_size_bytes": size_bytes,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX graph construction benchmarks")
    parser.add_argument("--json", type=str, help="Write results as JSON to this path")
    args = parser.parse_args()

    print("Running ONNX graph construction benchmarks...")
    print("  Iterations per model: 3")
    print(f"  Config: hidden={TINY_HIDDEN}, layers={TINY_LAYERS}, vocab={TINY_VOCAB}")
    results = run_benchmarks()
    if args.json:
        write_json(results, args.json)
        print(f"Results written to {args.json}")
    else:
        print_table(results)
