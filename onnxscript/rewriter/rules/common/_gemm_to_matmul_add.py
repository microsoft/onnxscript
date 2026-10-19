# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from onnxscript.rewriter._rewrite_rule import RewriteRule
from onnxscript.rewriter.rules.common._broadcast_to_matmul import check_if_not_need_reshape


# Pattern to match against
def reshape_gemm_reshape_pattern(op, input_a, input_b, input_c, shape_a, shape_c):
    reshape_a = op.Reshape(input_a, shape_a)
    # TODO: Temporary workaround to support benchmodels.
    # Tracked by https://github.com/microsoft/onnx-rewriter/issues/197.
    gemm = op.Gemm(reshape_a, input_b, input_c, alpha=1.0, beta=1.0)
    return op.Reshape(gemm, shape_c)


def matmul_add(op, input_a, input_b, input_c, **_):
    matmul = op.MatMul(input_a, input_b)
    return op.Add(matmul, input_c)


def check_if_not_need_reshape_for_gemm(context, *args, **kwargs):
    """Reject Gemm transposes because the MatMul replacement does not preserve them."""
    gemm = context.root.inputs[0].producer()
    if gemm is None:
        return False
    if gemm.attributes.get_int("transA", 0) != 0 or gemm.attributes.get_int("transB", 0) != 0:
        return False
    return check_if_not_need_reshape(context, *args, **kwargs)


gemm_to_matmul_add_rule = RewriteRule(
    reshape_gemm_reshape_pattern, matmul_add, check_if_not_need_reshape_for_gemm
)
