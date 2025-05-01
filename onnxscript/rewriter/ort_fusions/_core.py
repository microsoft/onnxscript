# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.ir.passes.common import shape_inference
from onnxscript.optimizer import optimize
from onnxscript.rewriter import rewrite
from onnxscript.rewriter.ort_fusions import (
    fused_matmul_rule_sets,
    # group_normalization_merge_silu,
    instance_to_group_normalization,
    softmax,
)
from onnxscript.rewriter.ort_fusions.attention import fuse_attention
from onnxscript.rewriter.ort_fusions.bias_gelu import fuse_bias_gelu
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.erfgelu import fuse_erfgelu
from onnxscript.rewriter.ort_fusions.fuse_packed_qkv_gqa import fuse_qkv_gqa
from onnxscript.rewriter.ort_fusions.gelu import fuse_gelu
from onnxscript.rewriter.ort_fusions.gqa import fuse_gqa
from onnxscript.rewriter.ort_fusions.mha import fuse_mha
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.ort_fusions.rotary_embedding import (
    fuse_partial_rotary_embedding,
    fuse_rotary_embedding,
)
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa
from onnxscript.rewriter.ort_fusions.skip_normalization import (
    fuse_skip_layer_normalization,
    fuse_skip_rms_normalization,
)

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
    # NOTE: group normalization merge silu should be applied after instance to group normalization
    # *group_normalization_merge_silu.rules.rules,
    *fused_matmul_rule_sets.fused_matmul_rule_sets(),
]


# Preliminary optimizations before applying the transformer fusions.
# TODO: There are some potential redundancies below. Can be targeted for optimization
# once we have robust fusion.
def _pre_optimize(model: ir.Model) -> ir.Model:
    # TODO: Do we need this dependence on ONNX's partial-data-propagation? There are some
    # extra shape-propagation and partial-data-propagation rules in ONNX that are not yet
    # incorporated in our optimizer.
    shape_inference.infer_shapes(model)
    optimize(model)
    return model


def fuse_xformers(model: ir.Model, debug: bool = False) -> tuple[ir.Model, dict[str, int]]:
    """
    Apply transformer-specific fusions to the given model.

    Args:
        model: The input ONNX model represented as an `ir.Model`.
        debug: If debug is True, enable pattern matching tracer for debugging.

    Returns:
        A tuple containing:
        - The optimized `ir.Model` after applying transformer-specific fusions.
        - A dictionary with a count of each of the fusions applied.
    """
    fusion_count = dict()

    model = _pre_optimize(model)

    def fuse(func, apply_shape_inference: bool = False):
        return func(model, debug=debug, apply_shape_inference=apply_shape_inference)

    fusion_count["erf_gelu"] = fuse(fuse_erfgelu)
    fusion_count["rms_normalization"] = fuse(fuse_rms_normalization)
    fusion_count["skip_layer_normalization"] = fuse(fuse_skip_layer_normalization)
    fusion_count["skip_rms_normalization"] = fuse(fuse_skip_rms_normalization)
    fusion_count["rotary_embedding"] = fuse(fuse_rotary_embedding)
    fusion_count["partial_rotary_embedding"] = fuse(fuse_partial_rotary_embedding)
    fusion_count["cos_sin_cache"] = fuse(fuse_cos_sin_cache)
    fusion_count["sdpa"] = fuse(fuse_sdpa, apply_shape_inference=True)
    # Optimize to avoid trying multiple attention-based fusions
    fusion_count["mha"] = fuse(fuse_mha)
    if fusion_count["mha"] == 0:
        # If no MHA fusion was applied, we can try the GQA fusion.
        # and avoid trying the attention fusion.
        fusion_count["gqa"] = fuse(fuse_gqa)
        fusion_count["packed_qkv_for_gqa"] = fuse(fuse_qkv_gqa)
        fusion_count["attention"] = 0
    else:
        fusion_count["attention"] = fuse(fuse_attention)
        fusion_count["gqa"] = 0
    fusion_count["gelu"] = fuse(fuse_gelu)
    fusion_count["bias_gelu"] = fuse(fuse_bias_gelu)
    # Finally: inline any intermediate fusion functions introduced that were not
    # consumed by other fusions, and eliminate any remaining unused nodes.
    optimize(model)
    return model, fusion_count


def optimize_for_ort(
    model: ir.Model,
    config_name: str | None = None,
    *,
    debug: bool = False,
) -> tuple[ir.Model, dict[str, int]]:
    """
    Optimize the model for ORT backend.

    TODO: config_name is not used yet. It should be used to select the appropriate
    optimization configuration (for an EP). Currently, a default implementation is used.

    Args:
        model: The model to optimize.
        config_name: The name of the configuration to use for optimization.
            Typically it identifies the Execution Provider (EP) to optimize for.
            If None, the default configuration will be used.
        debug: If debug is True, enable pattern matching tracer for debugging.

    Returns:
        A tuple containing:
        - The optimized `ir.Model` after applying transformer-specific fusions.
        - A dictionary with a count of each of the fusions applied.
    """

    model, fusion_count = fuse_xformers(
        model,
        debug=debug,
    )
    # Apply the ORT pattern rewrite rules.
    rewrite(model, ORT_PATTERN_REWRITE_RULES)
    return model, fusion_count
