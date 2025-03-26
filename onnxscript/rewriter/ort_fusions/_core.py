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
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.gelu import fuse_gelu
from onnxscript.rewriter.ort_fusions.mha import fuse_mha
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.ort_fusions.rotary_embedding import (
    fuse_partial_rotary_embedding,
    fuse_rotary_embedding,
)
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa
from onnxscript.rewriter.ort_fusions.skip_normalization import fuse_normalization

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
    model = shape_inference.infer_shapes(model)
    optimize(model)
    return model


def fuse_xformers(model: ir.Model) -> ir.Model:
    model = _pre_optimize(model)
    fuse_rms_normalization(model)
    fuse_normalization(model)
    fuse_rotary_embedding(model)
    fuse_partial_rotary_embedding(model)
    fuse_cos_sin_cache(model)
    fuse_sdpa(model)
    fuse_mha(model)
    fuse_gelu(model)
    # Finally: inline any intermediate fusion functions introduced that were not
    # consumed by other fusions, and eliminate any remaining unused nodes.
    optimize(model)
    return model


def optimize_for_ort(model: ir.Model, config_name: str | None = None) -> ir.Model:
    """
    Optimize the model for ORT backend.

    TODO: config_name is not used yet. It should be used to select the appropriate
    optimization configuration (for an EP). Currently, a default implementation is used.

    Args:
        model: The model to optimize.
        config_name: The name of the configuration to use for optimization.
            Typically it identifies the Execution Provider (EP) to optimize for.
            If None, the default configuration will be used.

    Returns:
        The optimized model.
    """

    fuse_xformers(model)
    rewrite(model, ORT_PATTERN_REWRITE_RULES)
    return model
