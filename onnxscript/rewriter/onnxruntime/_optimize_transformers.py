# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter.onnxruntime import attention, multi_head_attention, rms_normalization, rotary_embedding, skip_normalization
from onnxscript.rewriter import no_op
from onnxscript.optimizer import _constant_folding, remove_unused_nodes

def optimize(irmodel: ir.Model) -> None:

    def apply(rulename: str, rule):
        count = rule.apply_to_model(irmodel)
        print(f"{rulename} count: {count}")

    _constant_folding.fold_constants(irmodel, input_size_limit=5120000*4, output_size_limit=5120000*4)

    apply("Dropout", no_op.dropout_zero_rule)
    remove_unused_nodes(irmodel)

    apply("RMS Normalization", rms_normalization.rule)
    apply("Skip Normalization", skip_normalization.rule)

    _constant_folding.fold_constants(irmodel)
    remove_unused_nodes(irmodel)

    apply("Attention", attention.rule)
    apply("Rotate", rotary_embedding.rule)
    apply("Embed", rotary_embedding.embed_rule)
    apply("Multi-Head-Attention", multi_head_attention.rule)
