# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Compare benchmark results against baseline for regression detection.

Usage::

    python tests/benchmark_compare.py \\
        --current results.json \\
        --baseline tests/perf_baseline.json \\
        --output comparison.md

Exit code 0 = no blockers, 1 = blocker regression detected.
Only deterministic metrics (num_nodes, model_size_bytes) can trigger
exit code 1. Timing metrics are advisory only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Only deterministic metrics are baselined and compared.
# Timing/memory metrics (build_time_s, peak_memory_mb) are
# environment-dependent and excluded from regression detection.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "model_size_bytes": (0.05, 0.10),
    "num_nodes": (0.05, 0.10),
}

_GITHUB_REPO_URL = "https://github.com/onnxruntime/mobius"


def compare(current_path: str, baseline_path: str) -> tuple[str, bool]:
    """Compare current vs baseline. Returns (markdown, has_blocker)."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    rows: list[tuple[str, str, str, str, str, str]] = []
    has_blocker = False

    base_sha = baseline.get("_metadata", {}).get("commit", "unknown")
    head_sha = current.get("_metadata", {}).get("commit", "unknown")

    for model, metrics in sorted(current["models"].items()):
        base = baseline.get("models", {}).get(model, {})
        for metric in THRESHOLDS:
            curr_val = metrics.get(metric)
            base_val = base.get(metric)
            if curr_val is None or base_val is None or base_val == 0:
                continue
            delta_pct = (curr_val - base_val) / base_val
            warn_t, block_t = THRESHOLDS[metric]
            if delta_pct > block_t:
                status = "\U0001f534"  # red circle
                has_blocker = True
            elif delta_pct > warn_t:
                status = "\u26a0\ufe0f"  # warning
            elif delta_pct < -0.02:
                status = "\U0001f7e2"  # green circle
            else:
                status = "\u26aa"  # white circle
            rows.append(
                (
                    model,
                    metric,
                    _fmt(base_val, metric),
                    _fmt(curr_val, metric),
                    f"{delta_pct:+.1%}",
                    status,
                )
            )

    def _sha_link(sha: str) -> str:
        return f"[`{sha}`]({_GITHUB_REPO_URL}/commit/{sha})"

    md = "## Performance Comparison\n\n"
    md += f"Comparing {_sha_link(base_sha)} → {_sha_link(head_sha)}\n\n"
    md += "| Model | Metric | Baseline | Current | Delta | |\n"
    md += "|---|---|---:|---:|---:|---|\n"
    for model, metric, base_s, curr_s, delta_s, status in rows:
        md += f"| {model} | {metric} | {base_s} | {curr_s} | {delta_s} | {status} |\n"

    if has_blocker:
        md += "\n> Blocker regression detected in deterministic metrics.\n"
    elif any(r[5] == "\u26a0\ufe0f" for r in rows):
        md += "\n> Warning: minor regressions detected. Review flagged metrics.\n"
    else:
        md += "\n> No performance regressions.\n"

    return md, has_blocker


def _fmt(value: float | int, metric: str) -> str:
    if metric == "model_size_bytes":
        return f"{value / 1024:.0f} KB"
    if metric in ("num_nodes", "num_models"):
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compare benchmark results against baseline")
    p.add_argument("--current", required=True)
    p.add_argument("--baseline", default="tests/perf_baseline.json")
    p.add_argument("--output", default="comparison.md")
    args = p.parse_args()
    md, has_blocker = compare(args.current, args.baseline)
    Path(args.output).write_text(md)
    print(md)
    sys.exit(1 if has_blocker else 0)
