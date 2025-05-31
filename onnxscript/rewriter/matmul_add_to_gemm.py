# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Does the following transformation:
- MatMul ∘ Add -> Gemm
"""

from onnxscript.rewriter import pattern


def matmul_add(op, input_a, input_b, input_c):
    matmul = op.MatMul(input_a, input_b)
    return op.Add(matmul, input_c)


def gemm_pattern(op, input_a, input_b, input_c):
    return op.Gemm(input_a, input_b, input_c)


def check_shapes(*_, input_a, input_b, **__):
    # Rank of input_a and input_b must be 2
    return len(input_a.shape) == 2 and len(input_b.shape) == 2


rule = pattern.RewriteRule(matmul_add, gemm_pattern, check_shapes)
