# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-metric evaluation utilities for ONNX↔HF parity testing.

Provides ParityReport dataclass and level-appropriate comparison functions:
- compare_synthetic(): atol/rtol-gated (for L3 synthetic parity)
- compare_golden(): argmax-gated (for L4 golden comparison)
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np


class ParityResult(enum.Enum):
    """Outcome of a parity comparison."""

    PASS = "pass"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"


@dataclasses.dataclass
class ParityReport:
    """Multi-metric parity evaluation result.

    All metrics are always computed regardless of which gate is used.
    The ``result`` field reflects the level-specific gate decision.
    """

    # Primary metrics
    argmax_match: bool
    atol_pass: bool
    rtol_pass: bool
    max_abs_diff: float
    mean_abs_diff: float

    # Quality metrics
    top10_jaccard: float
    cosine_similarity: float

    # Near-tie detection
    top1_logit: float
    top2_logit: float
    near_tie: bool
    near_tie_margin: float

    # Overall result (depends on level)
    result: ParityResult
    level: str
    message: str


# Near-tie margins by dtype.  When |top1 - top2| < margin the token
# prediction is considered unstable and a mismatch is downgraded to
# AMBIGUOUS rather than FAIL.
NEAR_TIE_MARGINS: dict[str, float] = {
    "float32": 0.01,
    "float16": 0.1,
    "bfloat16": 0.5,
    "int4": 1.0,
}

# Per-dtype default tolerances for L3 synthetic parity.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # (atol, rtol)
    "float32": (1e-3, 1e-3),
    "float16": (1e-1, 1e-1),
    "bfloat16": (5e-1, 5e-1),
}


def _compute_metrics(
    onnx_logits: np.ndarray,
    hf_logits: np.ndarray,
    dtype: str,
) -> dict:
    """Compute all parity metrics between two logit tensors.

    Both inputs should be the last-token logits with shape
    ``(batch, vocab)`` or ``(vocab,)``.
    """
    # Flatten to last token if 3-D (batch, seq, vocab) → take last token
    if onnx_logits.ndim == 3:
        onnx_logits = onnx_logits[:, -1, :]
    if hf_logits.ndim == 3:
        hf_logits = hf_logits[:, -1, :]

    # Squeeze batch dim for single-sample comparison
    onnx_flat = onnx_logits.reshape(-1).astype(np.float64)
    hf_flat = hf_logits.reshape(-1).astype(np.float64)

    # Absolute difference metrics
    abs_diff = np.abs(onnx_flat - hf_flat)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Tolerance checks
    margin = NEAR_TIE_MARGINS.get(dtype, 0.01)
    atol_default, rtol_default = DEFAULT_TOLERANCES.get(dtype, (1e-3, 1e-3))

    # Use the last-token slice for argmax/top-k (first batch element)
    if onnx_logits.ndim == 2:
        onnx_last = onnx_logits[0]
        hf_last = hf_logits[0]
    else:
        onnx_last = onnx_logits
        hf_last = hf_logits

    onnx_top1 = int(np.argmax(onnx_last))
    hf_top1 = int(np.argmax(hf_last))
    argmax_match = onnx_top1 == hf_top1

    # Top-10 Jaccard similarity
    onnx_top10 = set(np.argsort(onnx_last)[-10:].tolist())
    hf_top10 = set(np.argsort(hf_last)[-10:].tolist())
    jaccard = len(onnx_top10 & hf_top10) / len(onnx_top10 | hf_top10)

    # Cosine similarity
    norm_onnx = np.linalg.norm(onnx_flat)
    norm_hf = np.linalg.norm(hf_flat)
    if norm_onnx > 0 and norm_hf > 0:
        cosine_sim = float(np.dot(onnx_flat, hf_flat) / (norm_onnx * norm_hf))
    else:
        cosine_sim = 0.0

    # Near-tie detection: are the top-2 HF logits very close?
    hf_sorted = np.sort(hf_last)[::-1]
    top1_logit = float(hf_sorted[0])
    top2_logit = float(hf_sorted[1]) if len(hf_sorted) > 1 else float("-inf")
    near_tie = abs(top1_logit - top2_logit) < margin

    return {
        "argmax_match": argmax_match,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "top10_jaccard": jaccard,
        "cosine_similarity": cosine_sim,
        "top1_logit": top1_logit,
        "top2_logit": top2_logit,
        "near_tie": near_tie,
        "near_tie_margin": margin,
        "atol_default": atol_default,
        "rtol_default": rtol_default,
    }


def compare_synthetic(
    onnx_logits: np.ndarray,
    hf_logits: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    dtype: str = "float32",
) -> ParityReport:
    """L3 synthetic parity comparison.  Gate: atol/rtol.

    With identical seeds and identical random weights, any atol
    divergence is a genuine op-level bug.  Argmax is reported as
    diagnostic only (noisy with random weights near zero).
    """
    assert onnx_logits.shape == hf_logits.shape, (
        f"Shape mismatch: ONNX {onnx_logits.shape} vs HF {hf_logits.shape}"
    )
    metrics = _compute_metrics(onnx_logits, hf_logits, dtype)

    atol_pass = bool(
        np.allclose(
            onnx_logits.astype(np.float64),
            hf_logits.astype(np.float64),
            atol=atol,
            rtol=0,
        )
    )
    rtol_pass = bool(
        np.allclose(
            onnx_logits.astype(np.float64),
            hf_logits.astype(np.float64),
            atol=0,
            rtol=rtol,
        )
    )

    # Gate: allclose (atol + rtol combined)
    allclose_pass = bool(
        np.allclose(
            onnx_logits.astype(np.float64),
            hf_logits.astype(np.float64),
            atol=atol,
            rtol=rtol,
        )
    )

    if allclose_pass:
        result = ParityResult.PASS
        message = (
            f"L3 PASS: max_abs_diff={metrics['max_abs_diff']:.6f}, "
            f"cosine={metrics['cosine_similarity']:.6f}"
        )
    else:
        result = ParityResult.FAIL
        message = (
            f"L3 FAIL: max_abs_diff={metrics['max_abs_diff']:.6f}, "
            f"mean_abs_diff={metrics['mean_abs_diff']:.6f}, "
            f"cosine={metrics['cosine_similarity']:.6f}, "
            f"argmax_match={metrics['argmax_match']}"
        )

    # Diagnostic: if near-tie detected, add a note
    if metrics["near_tie"]:
        message += " (near-tie: results may be seed-sensitive)"

    return ParityReport(
        argmax_match=metrics["argmax_match"],
        atol_pass=atol_pass,
        rtol_pass=rtol_pass,
        max_abs_diff=metrics["max_abs_diff"],
        mean_abs_diff=metrics["mean_abs_diff"],
        top10_jaccard=metrics["top10_jaccard"],
        cosine_similarity=metrics["cosine_similarity"],
        top1_logit=metrics["top1_logit"],
        top2_logit=metrics["top2_logit"],
        near_tie=metrics["near_tie"],
        near_tie_margin=metrics["near_tie_margin"],
        result=result,
        level="L3",
        message=message,
    )


def compare_golden(
    onnx_logits: np.ndarray,
    golden_top1_id: int,
    golden_top2_id: int,
    golden_top10_ids: list[int],
    golden_cosine_threshold: float = 0.9999,
    dtype: str = "float32",
) -> ParityReport:
    """L4 golden comparison.  Gate: argmax match (with near-tie AMBIGUOUS).

    With real weights, distributions are well-separated, so argmax
    is the meaningful gate.  Near-tie detection downgrades mismatches
    to AMBIGUOUS when |top1 - top2| < dtype-dependent margin.
    """
    margin = NEAR_TIE_MARGINS.get(dtype, 0.01)

    # Extract last-token logits
    if onnx_logits.ndim == 3:
        onnx_logits = onnx_logits[:, -1, :]
    if onnx_logits.ndim == 2:
        onnx_last = onnx_logits[0]
    else:
        onnx_last = onnx_logits

    onnx_last_f64 = onnx_last.astype(np.float64)
    onnx_top1 = int(np.argmax(onnx_last_f64))
    onnx_sorted = np.sort(onnx_last_f64)[::-1]
    top1_logit = float(onnx_sorted[0])
    top2_logit = float(onnx_sorted[1]) if len(onnx_sorted) > 1 else float("-inf")
    near_tie = abs(top1_logit - top2_logit) < margin

    argmax_match = onnx_top1 == golden_top1_id

    # Top-10 Jaccard
    onnx_top10 = set(np.argsort(onnx_last_f64)[-10:].tolist())
    golden_top10 = set(golden_top10_ids)
    jaccard = (
        len(onnx_top10 & golden_top10) / len(onnx_top10 | golden_top10)
        if golden_top10
        else 0.0
    )

    # Gate decision
    if argmax_match:
        result = ParityResult.PASS
        message = f"L4 PASS: argmax={onnx_top1}, top10_jaccard={jaccard:.2f}"
    elif near_tie and onnx_top1 == golden_top2_id:
        result = ParityResult.AMBIGUOUS
        message = (
            f"L4 AMBIGUOUS: argmax={onnx_top1} != golden_top1={golden_top1_id}, "
            f"but matches top2={golden_top2_id} and near-tie detected "
            f"(gap={abs(top1_logit - top2_logit):.4f} < margin={margin})"
        )
    else:
        result = ParityResult.FAIL
        message = (
            f"L4 FAIL: argmax={onnx_top1} != golden_top1={golden_top1_id}, "
            f"top10_jaccard={jaccard:.2f}"
        )

    return ParityReport(
        argmax_match=argmax_match,
        atol_pass=False,
        rtol_pass=False,
        max_abs_diff=0.0,
        mean_abs_diff=0.0,
        top10_jaccard=jaccard,
        cosine_similarity=0.0,
        top1_logit=top1_logit,
        top2_logit=top2_logit,
        near_tie=near_tie,
        near_tie_margin=margin,
        result=result,
        level="L4",
        message=message,
    )
