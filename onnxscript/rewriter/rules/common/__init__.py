# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
__all__ = [
    "add_0_rule",
    "affine_conv_fusion_rule",
    "cast_cast_rule",
    "cast_constant_of_shape_rule",
    "cast_constant_of_shape_without_value_rule",
    "collapse_slice_rule",
    "collapse_slice2_rule",
    "conv_affine_fusion_rule",
    "div_by_1_rule",
    "dropout_inference_rule",
    "dropout_zero_rule",
    "flatten_to_reshape_rule",
    "fuse_batchnorm_into_conv_rule",
    "fuse_batchnorm_into_conv_transpose_rule",
    "fuse_batchnorm_into_gemm_rule",
    "fuse_hardswish_rules",
    "fuse_pad_into_conv_integer_rule",
    "fuse_pad_into_conv_rule",
    "min_min_rule",
    "max_max_rule",
    "min_max_rule",
    "max_min_rule",
    "gemm_to_matmul_add_rule",
    "matmul_add_to_gemm_rule",
    "mul_by_1_rule",
    "no_op_cast_rule",
    "no_op_dynamic_scatter_nd_rule",
    "no_op_expand_rule",
    "no_op_static_scatter_nd_rule",
    "no_op_transpose_rule",
    "normalize_pad_format_conv_integer_rule",
    "normalize_pad_format_conv_rule",
    "one_reshape_matmul_reshape_rule",
    "reshape_reshape_rule",
    "slice_split_rule",
    "squeeze_reshape_1d_rule",
    "sub_0_rule",
    "successive_clip_relu_rule",
    "successive_clip_rule",
    "successive_relu_clip_rule",
    "successive_relu_rule",
    "transpose_a_matmul_add_to_gemm_rule",
    "transpose_ab_matmul_add_to_gemm_rule",
    "transpose_b_matmul_add_to_gemm_rule",
    "transpose_transpose_rule",
    "two_reshapes_matmul_reshape_rule",
    "unsqueeze_unsqueeze_rule",
]

from onnxscript.rewriter.rules.common._basic_rules import (
    cast_cast_rule,
    flatten_to_reshape_rule,
    no_op_cast_rule,
    no_op_expand_rule,
    no_op_transpose_rule,
    reshape_reshape_rule,
    slice_split_rule,
    squeeze_reshape_1d_rule,
    transpose_transpose_rule,
    unsqueeze_unsqueeze_rule,
)
from onnxscript.rewriter.rules.common._broadcast_to_matmul import (
    one_reshape_matmul_reshape_rule,
    two_reshapes_matmul_reshape_rule,
)
from onnxscript.rewriter.rules.common._cast_constant_of_shape import (
    cast_constant_of_shape_rule,
    cast_constant_of_shape_without_value_rule,
)
from onnxscript.rewriter.rules.common._collapse_slices import (
    collapse_slice2_rule,
    collapse_slice_rule,
)
from onnxscript.rewriter.rules.common._fuse_batchnorm import (
    fuse_batchnorm_into_conv_rule,
    fuse_batchnorm_into_conv_transpose_rule,
    fuse_batchnorm_into_gemm_rule,
)
from onnxscript.rewriter.rules.common._fuse_conv_affine import (
    affine_conv_fusion_rule,
    conv_affine_fusion_rule,
)
from onnxscript.rewriter.rules.common._fuse_hardswish import fuse_hardswish_rules
from onnxscript.rewriter.rules.common._fuse_pad_into_conv import (
    fuse_pad_into_conv_integer_rule,
    fuse_pad_into_conv_rule,
    normalize_pad_format_conv_integer_rule,
    normalize_pad_format_conv_rule,
)
from onnxscript.rewriter.rules.common._fuse_relus_clips import (
    successive_clip_relu_rule,
    successive_clip_rule,
    successive_relu_clip_rule,
    successive_relu_rule,
)
from onnxscript.rewriter.rules.common._gemm_to_matmul_add import gemm_to_matmul_add_rule
from onnxscript.rewriter.rules.common._matmul_add_to_gemm import (
    matmul_add_to_gemm_rule,
    transpose_a_matmul_add_to_gemm_rule,
    transpose_ab_matmul_add_to_gemm_rule,
    transpose_b_matmul_add_to_gemm_rule,
)
from onnxscript.rewriter.rules.common._min_max_to_clip import (
    max_max_rule,
    max_min_rule,
    min_max_rule,
    min_min_rule,
)
from onnxscript.rewriter.rules.common._no_op import (
    add_0_rule,
    div_by_1_rule,
    dropout_inference_rule,
    dropout_zero_rule,
    mul_by_1_rule,
    sub_0_rule,
)
from onnxscript.rewriter.rules.common._redundant_scatter_nd import (
    no_op_dynamic_scatter_nd_rule,
    no_op_static_scatter_nd_rule,
)
