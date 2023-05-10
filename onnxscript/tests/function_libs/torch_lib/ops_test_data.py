"""Test op correctness by comparing with PyTorch results.

## Usage

1. Set the env var CATCH_ORT_SEGFAULT to catch segfaults from ONNX Runtime.

## How to add a new operator test

This test use PyTorch's OpInfo mechanism to generate test cases for each operator.
You may find all OpInfos in https://github.com/pytorch/pytorch/blob/7ec0d6f006fdd2c9b978dc6aa4923144684a3f51/torch/testing/_internal/common_methods_invocations.py#L8804

1. To enable test cases for an operator
    1a. If the op is not `trace_only`, add an entry to the
    `OPINFO_FUNCTION_MAPPING_SCRIPTED` map.
    1b. If the op is `trace_only`, add an entry to the
    `OPINFO_FUNCTION_MAPPING_TRACE_ONLY` map.

    The entries are <op_info_name: function> pairs.
2. Edit `EXPECTED_SKIPS_OR_FAILS` and/or `SKIP_XFAIL_SUBTESTS` to skip or xfail tests.
Prefer xfail over skip when possible.
    2a. If a test is now failing because of xpass, because some previous errors
    are now fixed, removed the corresponding xfail.
3. If sample inputs of the OpInfo needs to be adjusted to fit the aten signature, create an input
wrangler function. See `_cat_input_wrangler` for an example.
4. To test different ONNX functions that are registered as overloads of the same
    op, use `ops_test_common.duplicate_opinfo` to create new OpInfo with new names and map each
    to one overload.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import definitions as opinfo_definitions

import onnxscript
import onnxscript.evaluator
from onnxscript._internal import version_utils
from onnxscript.function_libs.torch_lib.ops import core as core_ops
from onnxscript.function_libs.torch_lib.ops import nn as nn_ops
from onnxscript.function_libs.torch_lib.ops import special as special_ops
from onnxscript.tests.function_libs.torch_lib import extra_opinfo, ops_test_common

# For readability, these two are allowed to be imported given the high usage
from onnxscript.tests.function_libs.torch_lib.ops_test_common import skip, xfail

# Create a copy of the op_db to modify
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

# Append extra op_db into the op database for testing
OPS_DB.extend(opinfo_definitions.signal.op_db)
OPS_DB.extend(opinfo_definitions.special.op_db)
OPS_DB.extend(extra_opinfo.OP_DB)


# Modify this section ##########################################################


def _amin_amax_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "dim" not in kwargs:
        # Supply an empty dim to match the aten signature
        kwargs["dim"] = np.array([], dtype=np.int64)
    else:
        # Convert dim to a numpy array
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64).reshape((-1,))
    return args, kwargs


def _avg_pool2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "dim" not in kwargs:
        if len(args) > 6:
            kwargs["divisor_override"] = args.pop(6)
        if len(args) > 5:
            kwargs["count_include_pad"] = args.pop(5)
        if len(args) > 4:
            kwargs["ceil_mode"] = args.pop(4)
        if len(args) > 3:
            padding = args.pop(3)
            if isinstance(padding, np.ndarray):
                # Cannot using list(padding) here, because the element will be numpy.int64 instead of int
                padding = padding.tolist()
            kwargs["padding"] = padding
        stride = args.pop(2)
        if isinstance(stride, np.ndarray):
            stride = stride.tolist()
        kwargs["stride"] = stride
        kernel_size = args.pop(1)
        if isinstance(kernel_size, np.ndarray):
            kernel_size = kernel_size.tolist()
        kwargs["kernel_size"] = kernel_size
    return args, kwargs


def _cat_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Remove the self argument
    if len(args) == 2:
        kwargs["dim"] = args.pop()
    return args, kwargs


def _cross_entropy_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        reduction_vals = ["none", "mean", "sum"]
        value = kwargs["reduction"]
        idx = reduction_vals.index(value)
        kwargs["reduction"] = idx
    return args, kwargs


def _dropout_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "training" in kwargs:
        kwargs["train"] = kwargs["training"]
        kwargs.pop("training")
    return args, kwargs


def _embedding_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Remove arguments not present in the aten op signature."""
    if "max_norm" in kwargs:
        del kwargs["max_norm"]
    if "norm_type" in kwargs:
        del kwargs["norm_type"]
    return args, kwargs


def _empty_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Remove arguments not present in the aten op signature."""
    if "requires_grad" in kwargs:
        del kwargs["requires_grad"]
    return args, kwargs


def _flip_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the dims as tensor
    kwargs["dims"] = np.array(kwargs["dims"], dtype=np.int64)
    return args, kwargs


def _gather_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the dim argument an attribute
    kwargs["dim"] = args.pop(1)
    return args, kwargs


def _grid_sample_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Convert string attriute to int as input
    inter_mode_options = {"bilinear": 0, "nearest": 1, "bicubic": 2}
    padding_mode_options = {"zeros": 0, "border": 1, "reflection": 2}
    args.append(inter_mode_options[kwargs["mode"]])
    args.append(padding_mode_options[kwargs["padding_mode"]])
    args.append(kwargs["align_corners"])
    kwargs.clear()
    return args, kwargs


def _max_pool_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Remove return_indices argument because this op doesn't accept it
    if "return_indices" in kwargs:
        del kwargs["return_indices"]
    return args, kwargs


def _mse_loss_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        reduction_vals = ["none", "mean", "sum"]  # [0,1,2], default=1
        value = kwargs["reduction"]
        idx = reduction_vals.index(value)
        kwargs["reduction"] = idx
    return args, kwargs


def _nll_loss_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "reduction" in kwargs:
        # aten_nll_loss can only accept integer argument instead of string
        reduction_vals = ["none", "mean", "sum"]
        value = kwargs["reduction"]
        kwargs["reduction"] = reduction_vals.index(value)
    return args, kwargs


def _randn_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the size argument as attribute list[int]
    kwargs["size"] = args.pop(0).tolist()
    return args, kwargs


def _permute_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Change the dims argument back to a list because ONNX Transpose does not
    # support dynamic perms
    kwargs["dims"] = args.pop()
    kwargs["dims"] = kwargs["dims"].tolist()
    return args, kwargs


def _reflection_pad2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'reflect' arg
    return args, kwargs


def _replication_pad2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'replicate' arg
    return args, kwargs


def _replication_pad3d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args.pop(2)  # remove 'replicate' arg
    return args, kwargs


def _scatter_add_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    kwargs["dim"] = args.pop(1)
    return args, kwargs


def _scatter_reduce_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Put the string into kwargs, otherwise FullGraph mode will cannot find get 'reduce' argument
    kwargs["reduce"] = args.pop(4)
    return args, kwargs


def _sum_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if kwargs.get("dim") is not None:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
    return args, kwargs


def _upsample_bilinear2d_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "size" in kwargs:
        args.append(np.array(kwargs["size"], dtype=np.int64))
        del kwargs["size"]  # promote tensor type kwargs to args
    if "scale_factor" in kwargs:
        kwargs["scales_h"] = kwargs["scale_factor"]
        kwargs["scales_w"] = kwargs["scale_factor"]
        del kwargs["scale_factor"]  # adapt the function signature
    return args, kwargs


def _upsample_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "scale_factor" in kwargs:
        kwargs["scales_h"] = kwargs["scale_factor"]
        kwargs["scales_w"] = kwargs["scale_factor"]
        del kwargs["scale_factor"]
    if "size" in kwargs:
        kwargs["size"] = np.array(kwargs["size"], dtype=np.int64)
    return args, kwargs


def _unflatten_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    args[1] = np.array(args[1], dtype=np.int64)
    return args, kwargs


def _where_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # The aten::where op takes condition, x, y as inputs
    # Swap the first two inputs
    args[0], args[1] = args[1], args[0]
    return args, kwargs


# Ops to be tested for numerical consistency between onnx and pytorch
# Find the names of the OpInfos in torch/testing/_internal/common_methods_invocations.py

# Split the scripted and traced ops to make sure we don't forget to script an op
OPINFO_FUNCTION_MAPPING_SCRIPTED: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
    # onnxscript.OnnxFunction
    # | Callable[..., Any]
    # | tuple[
    #     onnxscript.OnnxFunction | Callable[..., Any],
    #     Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]],
    # ],
] = {
    "all_dim": core_ops.aten_all_dim,
    "allclose": core_ops.aten_allclose,
    "all": core_ops.aten_all,
    "abs": core_ops.aten_abs,
    "acos": core_ops.aten_acos,
    "acosh": core_ops.aten_acosh,
    "add": core_ops.aten_add,
    "addmm": core_ops.aten_addmm,
    # "alias": core_ops.aten_alias,  # alias is not in OP-TEST-DB
    "amax": (core_ops.aten_amax, _amin_amax_input_wrangler),
    "amin": (core_ops.aten_amin, _amin_amax_input_wrangler),
    "any": core_ops.aten_any,  # TODO: add more testcase which element is [0.0, 0.1, -0.1, 0.0] etc.
    "any_dim": core_ops.aten_any_dim,  # TODO: add more testcase which element is [0.0, 0.1, -0.1, 0.0] etc.
    "asin": core_ops.aten_asin,
    "asinh": core_ops.aten_asinh,
    "atan": core_ops.aten_atan,
    "atan2": core_ops.aten_atan2,
    "atanh": core_ops.aten_atanh,
    "baddbmm": core_ops.aten_baddbmm,
    "bmm": core_ops.aten_bmm,
    "broadcast_to": core_ops.aten_broadcast_to,
    "cat": (core_ops.aten_cat, _cat_input_wrangler),
    "ceil": core_ops.aten_ceil,
    "chunk": core_ops.aten_chunk,
    "clamp_max": core_ops.aten_clamp_max,
    "clamp_min": core_ops.aten_clamp_min,
    "clone": core_ops.aten_clone,
    "constant_pad_nd": core_ops.aten_constant_pad_nd,
    # "copy": core_ops.aten_copy,  # copy is not in OPS_DB
    "cos": core_ops.aten_cos,
    "cosh": core_ops.aten_cosh,
    "cross": core_ops.aten_cross,
    # "detach": core_ops.aten_detach,  # detach is not in OP-TEST-DB
    "div": core_ops.aten_div,
    "dot": core_ops.aten_dot,
    "empty": (core_ops.aten_empty, _empty_input_wrangler),
    # "empty_strided": core_ops.aten_empty_strided,  # empty_strided is not in OPS_DB
    "eq": core_ops.aten_eq,
    "equal": core_ops.aten_equal,
    "exp": core_ops.aten_exp,
    "exp2": core_ops.aten_exp2,
    "expand": core_ops.aten_expand,
    "expand_as": core_ops.aten_expand_as,
    "erf": core_ops.aten_erf,
    "fill": core_ops.aten_fill,
    "flip": (core_ops.aten_flip, _flip_input_wrangler),
    "floor": core_ops.aten_floor,
    "fmod": core_ops.aten_fmod,
    "full": core_ops.aten_full,
    "full_like": core_ops.aten_full_like,
    "gather": (core_ops.aten_gather, _gather_input_wrangler),
    "ge": core_ops.aten_ge,
    # "greater_equal": core_ops.aten_greater_equal,  # no test case in OPS_DB
    # "greater": core_ops.aten_greater,  # no test case in OPS_DB
    "gt": core_ops.aten_gt,
    # "is_same_size": core_ops.aten_is_same_size,  # no test case in OPS_DB
    # "is_nonzero": core_ops.aten_is_nonzero,  # no test case in OPS_DB
    "index_put_bool": core_ops.aten_index_put_bool,
    "index_put": core_ops.aten_index_put,
    "isclose": core_ops.aten_isclose,
    "isfinite": core_ops.aten_isfinite,
    "isinf": core_ops.aten_isinf,
    "isnan": core_ops.aten_isnan,
    "isneginf": core_ops.aten_isneginf,
    "isposinf": core_ops.aten_isposinf,
    "log": core_ops.aten_log,
    "le": core_ops.aten_le,
    "log10": core_ops.aten_log10,
    "log1p": core_ops.aten_log1p,
    "log_softmax": special_ops.aten_special_log_softmax,
    "log2": core_ops.aten_log2,
    "logaddexp": core_ops.aten_logaddexp,
    "logaddexp2": core_ops.aten_logaddexp2,
    "logcumsumexp": core_ops.aten_logcumsumexp,
    "logdet": core_ops.aten_logdet,
    "logsumexp": core_ops.aten_logsumexp,
    "lt": core_ops.aten_lt,
    "masked_fill": core_ops.aten_masked_fill,
    "matmul": core_ops.aten_matmul,
    "maximum": core_ops.aten_maximum,
    "min_dim": core_ops.aten_min_dim,
    "min_other": core_ops.aten_min_other,
    "min": core_ops.aten_min,
    "minimum": core_ops.aten_minimum,
    "mm": core_ops.aten_mm,
    "mul": core_ops.aten_mul,
    "narrow": core_ops.aten_narrow,
    # "native_dropout": core_ops.aten_native_dropout,  # native_dropout is not in OPS_DB
    "ne": core_ops.aten_ne,
    "neg": core_ops.aten_neg,
    "new_empty_dtype": core_ops.aten_new_empty_dtype,
    "new_empty": core_ops.aten_new_empty,
    "new_empty_strided_dtype": core_ops.aten_new_empty_strided_dtype,
    "new_empty_strided": core_ops.aten_new_empty_strided,
    "new_full_dtype": core_ops.aten_new_full_dtype,
    "new_full": core_ops.aten_new_full,
    "new_ones_dtype": core_ops.aten_new_ones_dtype,
    "new_ones": core_ops.aten_new_ones,
    "new_zeros_dtype": core_ops.aten_new_zeros_dtype,
    "new_zeros": core_ops.aten_new_zeros,
    "nn.functional.adaptive_avg_pool1d": nn_ops.aten_adaptive_avg_pool1d,
    "nn.functional.adaptive_avg_pool2d": nn_ops.aten_adaptive_avg_pool2d,
    "nn.functional.adaptive_avg_pool3d": nn_ops.aten_adaptive_avg_pool3d,
    "nn.functional.celu": nn_ops.aten_celu,
    # use cross_entropy as test case instead of cross_entropy_loss (not in OPS_DB)
    "nn.functional.cross_entropy": (
        nn_ops.aten_cross_entropy_loss,
        _cross_entropy_input_wrangler,
    ),
    "nn.functional.dropout": (core_ops.aten_dropout, _dropout_input_wrangler),
    "nn.functional.elu": nn_ops.aten_elu,
    "nn.functional.embedding": (core_ops.aten_embedding, _embedding_input_wrangler),
    "nn.functional.hardtanh": nn_ops.aten_hardtanh,
    "nn.functional.leaky_relu": nn_ops.aten_leaky_relu,
    "nn.functional.logsigmoid": nn_ops.aten_log_sigmoid,
    "nn.functional.nll_loss_weight": (nn_ops.aten_nll_loss_weight, _nll_loss_input_wrangler),
    "nn.functional.nll_loss": (nn_ops.aten_nll_loss, _nll_loss_input_wrangler),
    "nn.functional.reflection_pad2d": (
        nn_ops.aten_reflection_pad2d,
        _reflection_pad2d_input_wrangler,
    ),
    "nn.functional.relu": nn_ops.aten_relu,
    "nn.functional.relu6": nn_ops.aten_relu6,
    "nn.functional.replication_pad2d": (
        nn_ops.aten_replication_pad2d,
        _replication_pad2d_input_wrangler,
    ),
    "nn.functional.replication_pad3d": (
        nn_ops.aten_replication_pad3d,
        _replication_pad3d_input_wrangler,
    ),
    "nn.functional.selu": core_ops.aten_selu,
    "nn.functional.mse_loss": (nn_ops.aten_mse_loss, _mse_loss_input_wrangler),
    "nonzero": core_ops.aten_nonzero,
    "normal": core_ops.aten_normal,
    "ones": core_ops.aten_ones,
    "permute": (core_ops.aten_permute, _permute_input_wrangler),
    "pow": core_ops.aten_pow,
    # "rand": core_ops.aten_rand,  # no test case in OPS_DB
    "randn": (core_ops.aten_randn, _randn_input_wrangler),
    "reciprocal": core_ops.aten_reciprocal,
    "remainder": core_ops.aten_remainder,
    "repeat": core_ops.aten_repeat,
    "reshape": core_ops.aten_reshape,
    "resolve_conj": core_ops.aten_resolve_conj,
    "resolve_neg": core_ops.aten_resolve_neg,
    "round": core_ops.aten_round,
    "rsqrt": core_ops.aten_rsqrt,
    "rsub": core_ops.aten_rsub,
    "select": core_ops.aten_select,
    # "scalar_tensor": core_ops.aten_scalar_tensor,  # no test case in OPS_DB
    "scatter_add": (core_ops.aten_scatter_add, _scatter_add_input_wrangler),
    "sigmoid": core_ops.aten_sigmoid,
    "sign": core_ops.aten_sign,
    "sin": core_ops.aten_sin,
    "sinh": core_ops.aten_sinh,
    "softmax": special_ops.aten_special_softmax,
    "split_with_sizes": core_ops.aten_split_with_sizes,
    "split": core_ops.aten_split,
    "sqrt": core_ops.aten_sqrt,
    "squeeze_dim": core_ops.aten_squeeze_dim,
    "squeeze": core_ops.aten_squeeze,
    "stack": core_ops.aten_stack,
    "sub": core_ops.aten_sub,
    # "sym_size": core_ops.aten_sym_size,  # no test case in OPS_DB
    "t": core_ops.aten_t,
    "tan": core_ops.aten_tan,
    "tanh": core_ops.aten_tanh,
    "tile": core_ops.aten_tile,
    "topk": core_ops.aten_topk,
    "tril": core_ops.aten_tril,
    "triu": core_ops.aten_triu,
    "trunc": core_ops.aten_trunc,
    "unflatten": (core_ops.aten_unflatten, _unflatten_input_wrangler),
    "unsqueeze": core_ops.aten_unsqueeze,
    "view": core_ops.aten_view,
    "where": (core_ops.aten_where, _where_input_wrangler),
    "xlogy": special_ops.aten_special_xlogy,
    "zeros": core_ops.aten_zeros,
}


OPINFO_FUNCTION_MAPPING_TRACE_ONLY: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
] = {
    "arange_start_step": core_ops.aten_arange_start_step,
    "arange_start": core_ops.aten_arange_start,
    "arange": core_ops.aten_arange,
    "argmax": core_ops.aten_argmax,
    "argmin": core_ops.aten_argmin,
    "as_strided": core_ops.aten_as_strided,
    "clamp": core_ops.aten_clamp,
    "col2im": nn_ops.aten_col2im,
    "cumsum": core_ops.aten_cumsum,
    "contiguous": core_ops.aten_contiguous,
    "convolution": core_ops.aten_convolution,
    "empty_like": core_ops.aten_empty_like,
    "grid_sampler_2d": core_ops.aten_grid_sampler_2d,
    "nn.functional.grid_sample": (core_ops.aten_grid_sampler, _grid_sample_input_wrangler),
    "index_select": core_ops.aten_index_select,
    "layer_norm": core_ops.aten_layer_norm,
    "max": core_ops.aten_max,
    "max_pool2d": nn_ops.aten_max_pool2d,  # Custom from extra_opinfo
    "max_pool3d": nn_ops.aten_max_pool3d,  # Custom from extra_opinfo
    "native_batch_norm": core_ops.aten_native_batch_norm,
    "native_group_norm": core_ops.aten_native_group_norm,
    "native_layer_norm": core_ops.aten_native_layer_norm,
    "nn.functional.avg_pool2d": (nn_ops.aten_avg_pool2d, _avg_pool2d_input_wrangler),
    "nn.functional.conv1d": core_ops.aten_conv1d,
    "nn.functional.conv2d": core_ops.aten_conv2d,
    "nn.functional.conv3d": core_ops.aten_conv3d,
    "nn.functional.gelu": nn_ops.aten_gelu,
    "nn.functional.linear": nn_ops.aten_linear,
    "nn.functional.max_pool2d": (nn_ops.aten_max_pool2d, _max_pool_input_wrangler),
    "nn.functional.max_pool2d_with_indices": (
        nn_ops.aten_max_pool2d_with_indices,
        _max_pool_input_wrangler,
    ),
    "nn.functional.max_pool3d": (nn_ops.aten_max_pool3d, _max_pool_input_wrangler),
    "nn.functional.max_pool3d_with_indices": (
        nn_ops.aten_max_pool3d_with_indices,
        _max_pool_input_wrangler,
    ),
    "nn.functional.scaled_dot_product_attention": nn_ops.aten_scaled_dot_product_attention,
    "nn.functional.scaled_dot_product_attention_bool_mask": nn_ops.aten_scaled_dot_product_attention_bool_mask,
    "nn.functional.upsample_bilinear2d": (
        nn_ops.aten_upsample_bilinear2d,
        _upsample_bilinear2d_input_wrangler,
    ),
    "nn.functional.upsample_nearest2d": (
        nn_ops.aten_upsample_nearest2d,
        _upsample_input_wrangler,
    ),
    "ones_like": core_ops.aten_ones_like,
    "scatter_reduce": (core_ops.aten_scatter_reduce, _scatter_reduce_input_wrangler),
    "slice_scatter": core_ops.aten_slice_scatter,
    "slice": core_ops.aten_slice,
    "sum": (core_ops.aten_sum_dim_IntList, _sum_input_wrangler),
    "transpose": core_ops.aten_transpose,
    "var_mean": core_ops.aten_var_mean,
    "var_mean_dim": core_ops.aten_var_mean_dim,
    "var_mean_correction": core_ops.aten_var_mean_correction,
    "zeros_like": core_ops.aten_zeros_like,
}

# These ops are not deterministic, so we check shape and dtype only
NONDETERMINISTIC_OPS: frozenset[str] = frozenset(
    (
        "empty_like",
        "empty",
        "new_empty_strided_dtype",
        "new_empty_strided",
        "new_empty_dtype",
        "new_empty",
        "normal",
        "randn",
    )
)

OPINFO_FUNCTION_MAPPING: dict[
    str,
    onnxscript.OnnxFunction
    | Callable[..., Any]
    | tuple[
        onnxscript.OnnxFunction | Callable[..., Any],
        Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]],
    ],
] = {**OPINFO_FUNCTION_MAPPING_SCRIPTED, **OPINFO_FUNCTION_MAPPING_TRACE_ONLY}

TESTED_OPS = frozenset(OPINFO_FUNCTION_MAPPING)

EXPECTED_SKIPS_OR_FAILS = (
    xfail(
        "as_strided",
        variant_name="partial_views",
        reason="ONNX doesn't have partial view for tensor",
    ),
    xfail("logcumsumexp", reason="naive implementation not numerically stable"),
    xfail(
        "max",
        variant_name="binary",
        reason="fixme: current implementation gets shape inference error",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "max",
        variant_name="reduction_with_dim",
        reason="fixme: current implementation gets shape inference error",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "max_pool3d",
        variant_name="empty_strides",
        reason="fixme: 'shape' do not match: torch.Size([2, 3, 4, 3]) != torch.Size([2, 3, 4, 2])",
    ),
    xfail(
        "min_dim",
        variant_name="reduction_with_dim",
        reason="ORT Graph attribute inferencing failed https://github.com/onnx/onnx/issues/4986",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "new_empty_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_empty_dtype: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_empty_strided_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_empty_strided_dtype: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_empty_strided",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_empty_strided: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_full_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full_dtype: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_ones_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_ones_dtype: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_zeros_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_zeros_dtype: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool1d",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_adaptive_avg_pool1d: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "nn.functional.adaptive_avg_pool3d",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_adaptive_avg_pool3d: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        reason="fixme: ORT crashes on Windows, segfaults randomly on Linux",
    ),
    skip(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        reason="fixme: ORT crashes on Windows, segfaults randomly on Linux",
    ),
    xfail(
        "nn.functional.upsample_bilinear2d",
        reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "nn.functional.upsample_nearest2d",
        reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "repeat",
        reason="Shape inference error. Remove after ONNX 1.14 release",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnx_older_than("1.14"),
    ),
    xfail(
        "round",
        variant_name="decimals_0",
        reason="The op does not support decimals yet",
        test_class_name="TestOutputConsistencyEager",
    ),
    xfail("round", variant_name="decimals_3", reason="The op does not support decimals yet"),
    xfail(
        "round", variant_name="decimals_neg_3", reason="The op does not support decimals yet"
    ),
    xfail(
        "scatter_reduce",
        variant_name="mean",
        reason="ONNX doesn't support reduce='mean' option",
    ),
    xfail(
        "t",
        reason="ORT Graph attribute inferencing failed on rank-1 input",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "tile",
        reason="Shape inference error. Remove after ONNX 1.14 release",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnx_older_than("1.14")
        or version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "unflatten",
        reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
)


SKIP_XFAIL_SUBTESTS: tuple[ops_test_common.DecorateMeta, ...] = (
    skip(
        "all",
        matcher=lambda sample: not (len(sample.kwargs) == 0),
        reason="this Aten overload only support one tensor as input by design",
    ),
    xfail(
        "all_dim",
        matcher=lambda sample: not (len(sample.kwargs) > 0),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design",
    ),
    skip(
        "any",
        matcher=lambda sample: not (len(sample.kwargs) == 0),
        reason="this Aten overload only support one tensor as input by design",
    ),
    skip(
        "any_dim",
        matcher=lambda sample: not (len(sample.kwargs) > 0),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design",
    ),
    skip(
        "amax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme (core dump): ORT aborts on scalar inputs to ReduceMax-18",
    ),
    skip(
        "amin",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme (core dump): ORT aborts on scalar inputs to ReduceMin-18",
    ),
    xfail(
        "arange",
        matcher=lambda sample: len(sample.args) != 0,
        reason="arange overload takes single argument",
    ),
    xfail(
        "arange",
        matcher=lambda sample: sample.kwargs.get("end") is not None,
        reason="arange overload does not support positional 'end' argument",
    ),
    skip(
        "arange_start",
        matcher=lambda sample: len(sample.args) != 1,
        reason="arange_start overload takes two arguments (input, start)",
    ),
    xfail(
        "arange_start_step",
        matcher=lambda sample: len(sample.args) != 2,
        reason="arange_start_step overload takes three arguments (input, start, step)",
    ),
    skip(
        "cat",
        matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
        reason="cat does not support zero-dim tensors yet",
    ),
    skip(
        "div",
        matcher=lambda sample: sample.kwargs.get("rounding_mode") is not None,
        reason="rounding_mode is not yet supported",
    ),
    skip(
        "nn.functional.grid_sample",
        # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
        matcher=lambda sample: sample.kwargs.get("mode") == "bicubic"
        or len(sample.args[0].shape) != 4,
        reason="fixme: 'bicubic' mode in ORT implemented differently with Torch and only support 4D-tensor",
    ),
    skip(
        "grid_sampler_2d",
        # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
        matcher=lambda sample: sample.args[1] == 2,
        reason="fixme: 'bicubic' mode in ORT implemented differently with Torch",
    ),
    xfail(
        "index_put",
        matcher=lambda sample: not (sample.args[0][0].dtype == torch.int64),
        reason="this Aten overload only support tensor(int) as args",
    ),
    xfail(
        "index_put_bool",
        matcher=lambda sample: not (sample.args[0][0].dtype == torch.bool),
        reason="this Aten overload only support tensor(bool) as args",
    ),
    skip(
        "matmul",
        matcher=lambda sample: torch.numel(sample.input) == 0,
        reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
    ),
    skip(
        "min",  # aten_mean
        matcher=lambda sample: len(sample.args) > 0,
        reason="this ATen overload only supports one tensor as input by design",
    ),
    xfail(
        "min_other",  # aten_min_other(self, other)
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another tensor as args",
    ),
    xfail(
        "min_dim",  # aten_min_dim(self, dim)
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and not isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another int as args",
    ),
    skip(
        "new_empty",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        reason="this Aten overload only accept 2 inputs:(self, size)",
    ),
    skip(
        "new_empty_dtype",
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="this Aten overload must have 3 inputs:(self, size, dtype)",
    ),
    skip(
        "new_empty_strided",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        reason="this Aten overload only accept 3 inputs:(self, size, stride)",
    ),
    skip(
        "new_empty_strided_dtype",
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="this Aten overload must have 4 inputs:(self, size, stride, dtype)",
    ),
    skip(
        "new_full",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        reason="this Aten overload only accept 3 inputs:(self, size, fill_value)",
    ),
    skip(
        "new_full_dtype",
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="this Aten overload must have 4 inputs:(self, size, fill_value, dtype)",
    ),
    skip(
        "new_ones",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        reason="",
    ),
    skip(
        "new_ones_dtype",
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="",
    ),
    skip(
        "new_zeros",
        matcher=lambda sample: sample.kwargs.get("dtype") is not None,
        reason="",
    ),
    skip(
        "new_zeros_dtype",
        matcher=lambda sample: sample.kwargs.get("dtype") is None,
        reason="",
    ),
    xfail(
        "nonzero",
        matcher=lambda sample: sample.kwargs.get("as_tuple") is not None,
        reason="as_tuple=True is not supported",
    ),
    skip(
        "normal",
        matcher=lambda sample: len(sample.args) > 0 and not isinstance(sample.args[0], float),
        reason="ORT only accept float type for args[0] 'mean'",
    ),
    xfail(
        "nn.functional.adaptive_avg_pool1d",
        # Shape should be [N, C, D1]
        matcher=lambda sample: sample.args[0] not in {1, (1,)},
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    xfail(
        "nn.functional.adaptive_avg_pool2d",
        matcher=lambda sample: sample.args[0] != (1, 1),
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    xfail(
        "nn.functional.adaptive_avg_pool3d",
        matcher=lambda sample: sample.args[0] != (1, 1, 1),
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    xfail(
        "nn.functional.avg_pool2d",
        matcher=lambda sample: len(sample.args) > 5 and sample.args[5] is not None,
        reason="ONNX doesn't support divisor_override argument",
    ),
    xfail(
        "nn.functional.conv1d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv1d",
    ),
    xfail(
        "nn.functional.conv2d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv2d",
    ),
    xfail(
        "nn.functional.cross_entropy",
        matcher=lambda sample: not isinstance(sample.kwargs.get("weight"), int),
        reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[weight] is int type",
    ),
    skip(
        "nn.functional.dropout",
        matcher=lambda sample: len(sample.kwargs) == 0 or sample.kwargs.get("p", 0.0) > 0.0,
        reason="dropout is random so the result not match",
    ),
    skip(
        "nn.functional.max_pool2d_with_indices",
        matcher=lambda sample: sample.kwargs.get("return_indices") is False,
        reason="this aten overload assume return_indices=True",
    ),
    skip(
        "nn.functional.max_pool2d",
        matcher=lambda sample: sample.kwargs.get("return_indices") is True,
        reason="this aten overload assume return_indices=False",
    ),
    skip(
        "nn.functional.max_pool3d",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    ),
    skip(
        "nn.functional.max_pool3d",
        matcher=lambda sample: sample.kwargs.get("return_indices") is True,
        reason="this aten overload assume return_indices=False",
    ),
    skip(
        "nn.functional.max_pool3d_with_indices",
        matcher=lambda sample: sample.kwargs.get("ceil_mode") is True
        and sample.kwargs.get("padding") == 1,
        reason="FIXME: After https://github.com/microsoft/onnxruntime/issues/15446 is fixed",
    ),
    skip(
        "nn.functional.max_pool3d_with_indices",
        matcher=lambda sample: sample.kwargs.get("return_indices") is False,
        reason="this aten overload assume return_indices=True",
    ),
    skip(
        "nn.functional.nll_loss",
        matcher=lambda sample: "weight" in sample.kwargs,
        reason="this Aten overload doesn't accept weight as kwargs",
    ),
    skip(
        "nn.functional.nll_loss_weight",
        matcher=lambda sample: "weight" not in sample.kwargs,
        reason="this Aten overload need weight as kwargs",
    ),
    skip(
        "nn.functional.reflection_pad2d",
        matcher=lambda sample: not (len(sample.args) > 1 and sample.args[1] == "reflect"),
        reason="this Aten overload need args[1] == 'reflect' for pad mode",
    ),
    skip(
        "nn.functional.replication_pad2d",
        matcher=lambda sample: not (len(sample.args) > 1 and sample.args[1] == "replicate"),
        reason="this Aten overload need args[1] == 'replicate' for pad mode",
    ),
    skip(
        "nn.functional.replication_pad3d",
        matcher=lambda sample: not (
            len(sample.args) > 1
            and sample.args[1] == "replicate"
            and len(sample.input.shape) == 5
        ),
        reason="this Aten overload need args[1] == 'replicate' for pad mode, and 3D tensor",
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask")) is not None
        and attn_mask.dtype == torch.bool,
        reason="this overload takes a non-boolean mask",
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    ),
    skip(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask")) is not None
        and attn_mask.dtype != torch.bool,
        reason="this overload takes a boolean mask",
    ),
    skip(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
        reason="dropout is random so the results do not match",
    ),
    skip(
        "nn.functional.upsample_nearest2d",
        # Shape should be [N, C, H, W]
        matcher=lambda sample: len(sample.input.shape) != 2 + 2,
        reason="only test on 2d inputs",
    ),
    xfail(
        "nn.functional.upsample_nearest2d",
        matcher=lambda sample: "scale_factor" in sample.kwargs,
        reason="fixme: the scale_factor tests",
    ),
    xfail(
        "permute",
        matcher=lambda sample: len(list(filter(lambda v: v < 0, sample.args[0]))) > 0,
        reason="Negative value in perm is not supported",
    ),
    xfail(
        "permute",
        matcher=lambda sample: len(sample.args[0]) == 0,
        reason="Empty perm is not supported",
    ),
    xfail(
        "scatter_add",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch",
    ),
    skip(
        "scatter_reduce",
        # ONNX has not include_self parameter and default is include_self=True mode
        matcher=lambda sample: sample.kwargs.get("include_self") is False,
        reason="ONNX does't support include_self=False option",
    ),
    skip(
        "squeeze",
        matcher=lambda sample: not (len(sample.args) == 0),
        reason="this Aten overload only support one tensor as input by design",
    ),
    skip(
        "squeeze_dim",
        matcher=lambda sample: not (len(sample.args) > 0 and isinstance(sample.args[0], int)),
        reason="this Aten overload only support one tensor as input and one int as args by design",
    ),
    skip(
        "tile",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape)
        or not sample.input.shape,
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
    xfail(
        "unflatten",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
    xfail(
        "var_mean",
        # kwargs is empty
        matcher=lambda sample: len(sample.kwargs) > 0,
        reason="this Aten overload only support input[0]=tensor and input[1]=bool as input without any kwargs",
    ),
    xfail(
        "var_mean_dim",
        # kwargs["dim"] must exist, kwargs["correction"] must not exist
        matcher=lambda sample: not (
            sample.kwargs.get("dim", None) is not None
            and sample.kwargs.get("correction", None) is None
        ),
        reason="this Aten overload only support with 'dim' argument and without 'correction' argument",
    ),
    skip(
        "var_mean_correction",
        # Don't accept input[1]=bool and 'correction' must be in kwargs
        matcher=lambda sample: len(sample.args) > 0 or "correction" not in sample.kwargs,
        reason="this Aten overload only support when correction attribute exists",
    ),
)

ops_test_common.duplicate_opinfo(OPS_DB, "all", ("all_dim",))

ops_test_common.duplicate_opinfo(OPS_DB, "any", ("any_dim",))

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "arange",
    (
        "arange_start",
        "arange_start_step",
    ),
)

ops_test_common.duplicate_opinfo(OPS_DB, "index_put", ("index_put_bool",))

ops_test_common.duplicate_opinfo(OPS_DB, "new_empty", ("new_empty_dtype",))

ops_test_common.duplicate_opinfo(OPS_DB, "new_empty_strided", ("new_empty_strided_dtype",))

ops_test_common.duplicate_opinfo(OPS_DB, "new_full", ("new_full_dtype",))

ops_test_common.duplicate_opinfo(OPS_DB, "new_ones", ("new_ones_dtype",))

ops_test_common.duplicate_opinfo(OPS_DB, "new_zeros", ("new_zeros_dtype",))

ops_test_common.duplicate_opinfo(
    OPS_DB, "nn.functional.nll_loss", ("nn.functional.nll_loss_weight",)
)

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.pad",
    (
        "nn.functional.reflection_pad2d",
        "nn.functional.replication_pad2d",
        "nn.functional.replication_pad3d",
    ),
)

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.scaled_dot_product_attention",
    ("nn.functional.scaled_dot_product_attention_bool_mask",),
)

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "min",
    (
        "min_other",
        "min_dim",
    ),
)

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.upsample_bilinear",
    ("nn.functional.upsample_bilinear2d",),
)

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "nn.functional.upsample_nearest",
    (
        "nn.functional.upsample_nearest1d",
        "nn.functional.upsample_nearest2d",
        "nn.functional.upsample_nearest3d",
    ),
)

ops_test_common.duplicate_opinfo(OPS_DB, "squeeze", ("squeeze_dim",))

ops_test_common.duplicate_opinfo(
    OPS_DB,
    "var_mean",
    (
        "var_mean_dim",
        "var_mean_correction",
    ),
)

# NOTE: Complex supported functions
# TODO: Expand this list with trace_only_ops when it is needed
# Ops to be tested for numerical consistency between onnx and pytorch
# Find the names of the OpInfos in torch/testing/_internal/common_methods_invocations.py
COMPLEX_FUNCTION_MAPPING_SCRIPTED: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
    # onnxscript.OnnxFunction
    # | Callable[..., Any]
    # | tuple[
    #     onnxscript.OnnxFunction | Callable[..., Any],
    #     Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]],
    # ],
] = {
    "abs": core_ops.aten_abs_complex,
}

COMPLEX_TESTED_OPS = frozenset(COMPLEX_FUNCTION_MAPPING_SCRIPTED)

# Call dir(torch.ops.prims) and compare with entries in OPS_DB to create OpInfo for newly added prims ops
PRIMS_OPS_WITH_OP_INFO = (
    "abs",
    "acos",
    "acosh",
    "add",
    "amax",
    "amin",
    "as_strided",
    "as_strided_scatter",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cat",
    "ceil",
    "clone",
    "conj",
    "conj_physical",
    "cos",
    "cosh",
    "digamma",
    "div",
    "empty",
    "eq",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fill",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "full",
    "full_like",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "isfinite",
    "le",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "normal",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "reshape",
    "round",
    "rsqrt",
    "scalar_tensor",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "slice",
    "sqrt",
    "squeeze",
    "sub",
    "sum",
    "svd",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "uniform",
    "var",
    "where",
)

for op in PRIMS_OPS_WITH_OP_INFO:
    # Duplicate opinfo for prim ops. The new names all start with "prims_". E.g. "abs" -> "prims_abs".
    ops_test_common.duplicate_opinfo_for_prims(OPS_DB, op)

# Duplicate cases where the prims op name is different from the torch op name
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "i0", "bessel_i0")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.bessel_j0", "bessel_j0")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.bessel_j1", "bessel_j1")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.erfcx", "erfcx")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i0e", "bessel_i0e")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i1", "bessel_i1")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.i1e", "bessel_i1e")
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.ndtri", "ndtri")
ops_test_common.duplicate_opinfo_for_prims(
    OPS_DB, "special.spherical_bessel_j0", "spherical_bessel_j0"
)
ops_test_common.duplicate_opinfo_for_prims(OPS_DB, "special.zeta", "zeta")

OP_WITH_SKIPPED_XFAIL_SUBTESTS = frozenset(meta.op_name for meta in SKIP_XFAIL_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)
# Assert all ops in OPINFO_FUNCTION_MAPPING are in the OPS_DB
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB"
assert NONDETERMINISTIC_OPS.issubset(
    TESTED_OPS
), f"{NONDETERMINISTIC_OPS - TESTED_OPS} not in TESTED_OPS"

# temporary list for FLOAT16 dtype testing
OPINFO_FUNCTION_TARGET_DTYPE = (
    "log_softmax",
)
