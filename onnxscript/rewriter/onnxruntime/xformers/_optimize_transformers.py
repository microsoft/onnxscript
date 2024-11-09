# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.optimizer import fold_constants_ir, remove_unused_nodes
from onnxscript.rewriter.onnxruntime.xformers import (
    mha_rules,
    rms_normalization_rules,
    rotary_embedding_rules,
    sdpa_rules,
    skip_normalization_rules,
)


def optimize(irmodel: ir.Model, verbose: int = 0) -> None:
    def apply(rulename: str, rule):
        count = rule.apply_to_model(irmodel, verbose=verbose)
        print(f"{rulename} count: {count}")

    fold_constants_ir(irmodel, input_size_limit=5120000 * 4, output_size_limit=5120000 * 4)
    remove_unused_nodes(irmodel)

    apply("RMS Normalization", rms_normalization_rules)
    apply("Skip Normalization", skip_normalization_rules)

    fold_constants_ir(irmodel)
    remove_unused_nodes(irmodel)

    apply("SDPA-Attention", sdpa_rules)
    apply("RotaryEmbedding", rotary_embedding_rules)
    apply("Multi-Head-Attention", mha_rules)
