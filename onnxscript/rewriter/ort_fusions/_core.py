# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import os
import tempfile

import onnx

import onnxscript.ir as ir
from onnxscript.optimizer import optimize, remove_unused_nodes
from onnxscript.rewriter import rewrite
from onnxscript.rewriter.llama_rule_sets import llama_p0_rule_set
from onnxscript.rewriter.ort_fusions import (
    fused_matmul_rule_sets,
    # group_normalization_merge_silu,
    instance_to_group_normalization,
    softmax,
)
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.mha import fuse_mha
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.ort_fusions.rotary_embedding import fuse_rotary_embedding
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa
from onnxscript.rewriter.ort_fusions.skip_normalization import fuse_normalization

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
    # NOTE: group normalization merge silu should be applied after instance to group normalization
    # *group_normalization_merge_silu.rules.rules,
    *fused_matmul_rule_sets.fused_matmul_rule_sets(),
]

_extra_opt_rules = llama_p0_rule_set()


# Preliminary optimizations before applying the transformer fusions.
# TODO: There are some potential redundancies below. Can be targeted for optimization
# once we have robust fusion.
def _pre_optimize(model):
    optimize(model)
    # TODO: Do we need this dependence on ONNX's partial-data-propagation? There are some
    # extra shape-propagation and partial-data-propagation rules in ONNX that are not yet
    # incorporated in our optimizer.
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "in.onnx")
        outfile = os.path.join(tmpdir, "out.onnx")
        ir.save(model, infile)
        onnx.shape_inference.infer_shapes_path(infile, outfile, True, True, True)
        model = ir.load(outfile)
    optimize(model)
    # TODO: The extra optimization rules should be included in the base optimization
    _extra_opt_rules.apply_to_model(model)
    optimize(model)
    return model


def fuse_xformers(model: ir.Model) -> None:
    _pre_optimize(model)
    fuse_rms_normalization(model)
    fuse_normalization(model)
    fuse_rotary_embedding(model)
    fuse_cos_sin_cache(model)
    fuse_sdpa(model)
    fuse_mha(model)
    remove_unused_nodes(model)


def optimize_for_ort(model: ir.Model) -> None:
    fuse_xformers(model)
    rewrite(model, ORT_PATTERN_REWRITE_RULES)
