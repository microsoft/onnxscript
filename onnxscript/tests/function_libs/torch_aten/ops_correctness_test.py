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
2. Edit `EXPECTED_SKIPS_OR_FAILS` and/or `SKIP_SUBTESTS` to skip or xfail tests.
Prefer xfail over skip when possible.
    2a. If a test is now failing because of xpass, because some previous errors
    are now fixed, removed the corresponding xfail.
3. If sample inputs of the OpInfo needs to be adjusted to fit the aten signature, create an input
wrangler function. See `_cat_input_wrangler` for an example.
4. To test different ONNX functions that are registered as overloads of the same
    op, use `duplicate_opinfo` to create new OpInfo with new names and map each
    to one overload.
"""
from __future__ import annotations

import copy
import dataclasses
import multiprocessing
import os
import pprint
import unittest
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state
import parameterized
import torch
from torch.testing._internal import common_device_type, common_methods_invocations
from torch.testing._internal.opinfo import core as opinfo_core
from torch.utils import _pytree as pytree

import onnxscript
import onnxscript.evaluator
from onnxscript._internal import version_utils
from onnxscript.function_libs.torch_aten import graph_building
from onnxscript.function_libs.torch_aten.ops import core as core_ops
from onnxscript.function_libs.torch_aten.ops import nn as nn_ops
from onnxscript.function_libs.torch_aten.ops import special as special_ops
from onnxscript.tests.function_libs.torch_aten import extra_opinfo

T = TypeVar("T")

# Test only float32 inputs. All dtypes will be tested on the generated symbolic functions.
TESTED_DTYPES = (torch.float32,)

# Convenience tuples for creating dtype lists when skipping or xfailing tests

BOOL_TYPES = (torch.bool,)

INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)

FLOAT_TYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
)

TEST_OPSET_VERSION = 18
IS_WINDOWS = os.name == "nt"


def dtypes_except(*dtypes: torch.dtype) -> Sequence[torch.dtype]:
    """Returns all dtypes except the ones specified."""
    return tuple(dtype for dtype in TESTED_DTYPES if dtype not in dtypes)


@dataclasses.dataclass
class DecorateMeta:
    """A dataclass for storing information about a test case to skip or xfail.

    Adapted from functorch: functorch/test/common_utils.py
    """

    op_name: str
    variant_name: str
    decorator: Callable[..., Any]
    dtypes: Optional[Collection[torch.dtype]]
    reason: str
    matcher: Optional[Callable[[Any], bool]] = None
    enabled_if: bool = True
    # The test_class_name to apply the decorator to. If None, the decorator is
    # applied to all test classes.
    test_class_name: Optional[str] = None


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
    enabled_if: bool = True,
    test_class_name: Optional[str] = None,
) -> DecorateMeta:
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for the failure.
        dtypes: The dtypes to expect the failure.
        enabled_if: Whether the xfail is enabled.
        test_class_name: The test class name to apply the xfail to. If None, the
            xfail is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        dtypes=dtypes,
        reason=reason,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    test_class_name: Optional[str] = None,
) -> DecorateMeta:
    """Skips an OpInfo test.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for skipping.
        dtypes: The dtypes to skip.
        matcher: A function that matches the test sample input. It is used only when
            the skip is in the SKIP_SUBTESTS list.
        enabled_if: Whether the skip is enabled.
        test_class_name: The test class name to apply the skip to. If None, the skip
            is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
    )


def add_decorate_info(
    all_opinfos: Sequence[opinfo_core.OpInfo],
    test_class_name: str,
    base_test_name: str,
    skip_or_xfails: Iterable[DecorateMeta],
) -> Callable[[T], T]:
    """Decorates OpInfo tests with decorators based on the skip_or_xfails list."""
    ops_mapping = {(info.name, info.variant_test_name): info for info in all_opinfos}
    for decorate_meta in skip_or_xfails:
        opinfo = ops_mapping.get((decorate_meta.op_name, decorate_meta.variant_name))
        assert (
            opinfo is not None
        ), f"Couldn't find OpInfo for {decorate_meta}. Did you need to specify variant_name?"
        decorators = list(opinfo.decorators)
        new_decorator = opinfo_core.DecorateInfo(
            decorate_meta.decorator,
            decorate_meta.test_class_name or test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
            active_if=decorate_meta.enabled_if,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def duplicate_opinfo(opinfos: list[opinfo_core.OpInfo], name: str, new_names: tuple[str, ...]):
    """Duplicate an opinfo in the opinfo database and give it a new name."""
    duplicated = []
    all_info_names = {opinfo.name for opinfo in opinfos}
    for opinfo in opinfos:
        if opinfo.name == name:
            for new_name in new_names:
                if new_name in all_info_names:
                    # NOTE: Avoid duplicating an opinfo that already exists in the database.
                    # New opinfos are expected to be added in torch-nightly.
                    warnings.warn(
                        f"OpInfo {new_name} already exists in the database.", stacklevel=1
                    )
                    continue
                new_opinfo = copy.deepcopy(opinfo)
                new_opinfo.name = new_name
                duplicated.append(new_opinfo)
    opinfos.extend(duplicated)


# Create a copy of the op_db to modify
OPS_DB = copy.deepcopy(common_methods_invocations.op_db)

# Append extra op_db into the op database for testing
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
    onnxscript.OnnxFunction
    | Callable[..., Any]
    | tuple[
        onnxscript.OnnxFunction | Callable[..., Any],
        Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]],
    ],
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
    "asin": core_ops.aten_asin,
    "asinh": core_ops.aten_asinh,
    "atan": core_ops.aten_atan,
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
    "any": core_ops.aten_any,  # TODO: add more testcase which element is [0.0, 0.1, -0.1, 0.0] etc.
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
    "index_select": core_ops.aten_index_select,
    "layer_norm": core_ops.aten_layer_norm,
    "max": core_ops.aten_max,
    "native_layer_norm": core_ops.aten_native_layer_norm,
    "new_empty": core_ops.aten_new_empty,
    "new_empty_strided": core_ops.aten_new_empty_strided,
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
    "slice": core_ops.aten_slice,
    "sum": (core_ops.aten_sum_dim_IntList, _sum_input_wrangler),
    "transpose": core_ops.aten_transpose,
    "zeros_like": core_ops.aten_zeros_like,
}

# These ops are not deterministic, so we check shape and dtype only
NONDETERMINISTIC_OPS: frozenset[str] = frozenset(
    (
        "empty_like",
        "empty",
        "new_empty_strided",
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
        "any",
        reason="fixme: ORT shape inference error",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "as_strided",
        variant_name="partial_views",
        reason="ONNX doesn't have partial view for tensor",
    ),
    xfail(
        "chunk", reason="fixme: ORT error", test_class_name="TestOutputConsistencyFullGraph"
    ),
    xfail(
        "index_select",
        reason="fixme: ORT shape inference error on rank-0 input",
        test_class_name="TestOutputConsistencyFullGraph",
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
        "min_dim",
        variant_name="reduction_with_dim",
        reason="ORT Graph attribute inferencing failed https://github.com/onnx/onnx/issues/4986",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
    xfail(
        "new_full",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_ones",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_ones_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_zeros",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full: failed validating the check: !(it.GetName().empty())'",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnxruntime_older_than("1.15"),
    ),
    xfail(
        "new_zeros_dtype",
        reason="fixme: ORT fails with invalid model: 'ONNX Schema aten_new_full: failed validating the check: !(it.GetName().empty())'",
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
    xfail(
        "nn.functional.mse_loss",
        reason="Shape inference error. Remove after ONNX 1.14 release",
        test_class_name="TestOutputConsistencyFullGraph",
        enabled_if=version_utils.onnx_older_than("1.14"),
    ),
    skip(
        "nn.functional.scaled_dot_product_attention",
        reason="fixme: ORT crashes on Windows",
        enabled_if=IS_WINDOWS,
    ),
    skip(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        reason="fixme: ORT crashes on Windows",
        enabled_if=IS_WINDOWS,
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
        "unflatten",
        reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
        test_class_name="TestOutputConsistencyFullGraph",
    ),
)


SKIP_SUBTESTS: tuple[DecorateMeta, ...] = (
    skip(
        "all",
        matcher=lambda sample: not (len(sample.kwargs) == 0),
        reason="this Aten overload only support one tensor as input by design",
    ),
    skip(
        "all_dim",
        matcher=lambda sample: not (len(sample.kwargs) > 0),
        reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design",
    ),
    skip(
        "amax",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: ORT aborts on scalar inputs to ReduceMax-18",
    ),
    skip(
        "amin",
        matcher=lambda sample: len(sample.input.shape) == 0,
        reason="fixme: ORT aborts on scalar inputs to ReduceMin-18",
    ),
    skip(
        "arange",
        matcher=lambda sample: len(sample.args) != 0,
        reason="arange overload takes single argument",
    ),
    skip(
        "arange",
        matcher=lambda sample: sample.kwargs.get("end") is not None,
        reason="arange overload does not support positional 'end' argument",
    ),
    skip(
        "arange_start",
        matcher=lambda sample: len(sample.args) != 1,
        reason="arange_start overload takes two arguments (input, start)",
    ),
    skip(
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
        "index_put",
        matcher=lambda sample: not (sample.args[0][0].dtype == torch.int64),
        reason="this Aten overload only support tensor(int) as args",
    ),
    skip(
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
    skip(
        "min_other",  # aten_min_other(self, other)
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another tensor as args",
    ),
    skip(
        "min_dim",  # aten_min_dim(self, dim)
        matcher=lambda sample: len(sample.args) == 0
        or (len(sample.args) > 0 and not isinstance(sample.args[0], int)),
        reason="this ATen overload only support one tensor as input and another int as args",
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
    skip(
        "nonzero",
        matcher=lambda sample: sample.kwargs.get("as_tuple") is not None,
        reason="as_tuple=True is not supported",
    ),
    skip(
        "normal",
        matcher=lambda sample: len(sample.args) > 0 and not isinstance(sample.args[0], float),
        reason="ORT only accept float type for args[0] 'mean'",
    ),
    skip(
        "nn.functional.adaptive_avg_pool1d",
        # Shape should be [N, C, D1]
        matcher=lambda sample: sample.args[0] not in {1, (1,)},
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    skip(
        "nn.functional.adaptive_avg_pool2d",
        matcher=lambda sample: sample.args[0] != (1, 1),
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    skip(
        "nn.functional.adaptive_avg_pool3d",
        matcher=lambda sample: sample.args[0] != (1, 1, 1),
        reason="only global pooling is supported; only batched inputs are supported",
    ),
    skip(
        "nn.functional.avg_pool2d",
        matcher=lambda sample: len(sample.args) > 5 and sample.args[5] is not None,
        reason="ONNX doesn't support divisor_override argument",
    ),
    skip(
        "nn.functional.conv1d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv1d",
    ),
    skip(
        "nn.functional.conv2d",
        matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
        reason="String padding is not accepted by aten::conv2d",
    ),
    skip(
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
    skip(
        "nn.functional.upsample_nearest2d",
        matcher=lambda sample: "scale_factor" in sample.kwargs,
        reason="fixme: the scale_factor tests",
    ),
    skip(
        "permute",
        matcher=lambda sample: len(list(filter(lambda v: v < 0, sample.args[0]))) > 0,
        reason="Negative value in perm is not supported",
    ),
    skip(
        "permute",
        matcher=lambda sample: len(sample.args[0]) == 0,
        reason="Empty perm is not supported",
    ),
    skip(
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
        "unflatten",
        matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
        reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
    ),
)

duplicate_opinfo(OPS_DB, "all", ("all_dim",))

duplicate_opinfo(
    OPS_DB,
    "arange",
    (
        "arange_start",
        "arange_start_step",
    ),
)

duplicate_opinfo(OPS_DB, "index_put", ("index_put_bool",))

duplicate_opinfo(OPS_DB, "new_ones", ("new_ones_dtype",))

duplicate_opinfo(OPS_DB, "new_zeros", ("new_zeros_dtype",))

duplicate_opinfo(OPS_DB, "nn.functional.nll_loss", ("nn.functional.nll_loss_weight",))

duplicate_opinfo(
    OPS_DB,
    "nn.functional.pad",
    (
        "nn.functional.reflection_pad2d",
        "nn.functional.replication_pad2d",
        "nn.functional.replication_pad3d",
    ),
)

duplicate_opinfo(
    OPS_DB,
    "nn.functional.scaled_dot_product_attention",
    ("nn.functional.scaled_dot_product_attention_bool_mask",),
)

duplicate_opinfo(
    OPS_DB,
    "min",
    (
        "min_other",
        "min_dim",
    ),
)

duplicate_opinfo(
    OPS_DB,
    "nn.functional.upsample_bilinear",
    ("nn.functional.upsample_bilinear2d",),
)

duplicate_opinfo(
    OPS_DB,
    "nn.functional.upsample_nearest",
    (
        "nn.functional.upsample_nearest1d",
        "nn.functional.upsample_nearest2d",
        "nn.functional.upsample_nearest3d",
    ),
)

duplicate_opinfo(OPS_DB, "squeeze", ("squeeze_dim",))


# END OF SECTION TO MODIFY #####################################################


OP_WITH_SKIPPED_SUBTESTS = frozenset(meta.op_name for meta in SKIP_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)
# Assert all ops in OPINFO_FUNCTION_MAPPING are in the OPS_DB
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB"
assert NONDETERMINISTIC_OPS.issubset(
    TESTED_OPS
), f"{NONDETERMINISTIC_OPS - TESTED_OPS} not in TESTED_OPS"

TORCH_TYPE_TO_ONNX = {
    torch.bool: onnx.TensorProto.BOOL,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.int8: onnx.TensorProto.INT8,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.complex64: onnx.TensorProto.COMPLEX64,
    torch.complex128: onnx.TensorProto.COMPLEX128,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}


def _convert_tensor_to_numpy(input: Any) -> Any:
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if isinstance(input[0], torch.Tensor):
            return [_convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)

    return input


def _convert_kwargs_for_onnx(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Converts kwargs to be compatible with ONNX Runtime.

    ONNX Runtime doesn't support torch.bool, so we convert them to torch.uint8.
    """
    new_kwargs = {}
    for key, value in kwargs.items():
        if key == "device":
            continue
        if key == "dtype":
            value = TORCH_TYPE_TO_ONNX[value]
        if isinstance(value, torch.Tensor):
            value = np.array(value)
        new_kwargs[key] = value
    return new_kwargs


def _should_skip_test_sample(op_name: str, sample) -> Optional[str]:
    """Returns a reason if a test sample should be skipped."""
    if op_name not in OP_WITH_SKIPPED_SUBTESTS:
        return None
    for decorator_meta in SKIP_SUBTESTS:
        # Linear search on SKIP_SUBTESTS. That's fine because the list is small.
        if decorator_meta.op_name == op_name:
            assert decorator_meta.matcher is not None, "Matcher must be defined"
            if decorator_meta.matcher(sample):
                return decorator_meta.reason
    return None


class OrtAbortedError(RuntimeError):
    """ONNX Runtime Aborted."""


def _ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    """Run a model with ONNX Runtime."""

    # Disable all ORT optimizations
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(serialized_model, session_options)
    return session.run(None, ort_inputs)


def _ort_session_run_return_dict(
    serialized_model: bytes, ort_inputs: Mapping[str, Any], return_dict
) -> None:
    """Run a model with ONNX Runtime and store the results in return_dict."""

    try:
        return_dict["results"] = _ort_session_run(serialized_model, ort_inputs)
        return_dict["error"] = None
    except Exception as e:  # pylint: disable=broad-except
        return_dict["results"] = None
        return_dict["error"] = e


def _safe_ort_session_run(serialized_model: bytes, ort_inputs: Mapping[str, Any]):
    """Run a model with ONNX Runtime in a separate process.

    Args:
        serialized_model: Serialized ONNX model proto.
        ort_inputs: Inputs to the model.

    Returns:
        The inference result.

    Raises:
        OrtAbortedError if the process did not execute successfully.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(
        target=_ort_session_run_return_dict, args=(serialized_model, ort_inputs, return_dict)
    )
    process.start()
    process.join()
    process.close()
    if not return_dict:
        raise OrtAbortedError()
    if return_dict["error"] is not None:
        raise return_dict["error"]
    return return_dict["results"]


def _format_model_and_input_information(onnx_model, inputs):
    return (
        f"Inputs:\n"
        f"{pprint.pformat(inputs)}\n"
        f"Model:\n"
        f"{onnxscript.proto2text(onnx_model)}"
    )


class TestFunctionValidity(unittest.TestCase):
    def test_all_script_functions_are_onnx_functions(self):
        functions = set()
        for func_with_wrangler in OPINFO_FUNCTION_MAPPING_SCRIPTED.values():
            if isinstance(func_with_wrangler, tuple):
                func = func_with_wrangler[0]
            else:
                func = func_with_wrangler
            functions.add(func)

        # TODO(justinchuby): Add from the registry
        for func in functions:
            if not isinstance(func, onnxscript.OnnxFunction):
                raise AssertionError(
                    f"'{func}' is not an OnnxFunction. Was it decorated with '@torch_op'? "
                    "If the function is trace_only, please move it to the "
                    "'OPINFO_FUNCTION_MAPPING_TRACE_ONLY' dict."
                )

    def test_all_trace_only_functions_are_not_onnx_functions(self):
        for func_with_wrangler in OPINFO_FUNCTION_MAPPING_TRACE_ONLY.values():
            if isinstance(func_with_wrangler, tuple):
                func = func_with_wrangler[0]
            else:
                func = func_with_wrangler
            if isinstance(func, onnxscript.OnnxFunction):
                raise AssertionError(
                    f"'{func.name}' is an OnnxFunction. "
                    "If the function is not trace_only, please move it to the "
                    "'OPINFO_FUNCTION_MAPPING_SCRIPTED' dict."
                )

    @parameterized.parameterized.expand(list(OPINFO_FUNCTION_MAPPING_SCRIPTED.items()))
    @unittest.skipIf(
        version_utils.onnx_older_than("1.14"),
        "Function checker is not available before ONNX 1.14",
    )
    def test_script_function_passes_checker(self, _, func_with_wrangler):
        if isinstance(func_with_wrangler, tuple):
            func = func_with_wrangler[0]
        else:
            func = func_with_wrangler
        function_proto = func.to_function_proto()
        onnx.checker.check_function(function_proto)  # type: ignore[attr-defined]


def _graph_executor(
    outputs: Sequence[Any],
) -> Callable[[Callable[..., Any], tuple[Any], dict[str, Any]], None]:
    """Eagerly executes a function."""

    def _capture_graph_and_evaluate_torch_script_evaluator(function: Callable, args, kwargs):
        """Captures the graph of a function and evaluates it using TorchScriptEvaluator."""

        # Initialize the ONNX graph
        onnxscript_graph = graph_building.TorchScriptGraph()
        tracer = graph_building.TorchScriptTracingEvaluator(onnxscript_graph)
        ort_inputs = {}
        onnxscript_args: list[Any] = []
        onnxscript_kwargs = {}
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                input_name = f"input_{i}"
                input = onnxscript_graph.add_input(
                    input_name,
                    torch.tensor(arg).shape,
                    torch.tensor(arg).dtype,
                )
                input.value = arg
                onnxscript_args.append(input)
                ort_inputs[input_name] = arg
            elif isinstance(arg, Sequence):
                sequence_input = []
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, np.ndarray):
                        input_name = f"input_{i}_{j}"
                        input = onnxscript_graph.add_input(
                            input_name,
                            torch.tensor(subarg).shape,
                            torch.tensor(subarg).dtype,
                        )
                        input.value = subarg
                        sequence_input.append(input)
                        ort_inputs[input_name] = subarg
                onnxscript_args.append(sequence_input)
            else:
                onnxscript_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                input = onnxscript_graph.add_input(
                    key,
                    torch.tensor(value).shape,
                    torch.tensor(value).dtype,
                )
                input.value = value
                ort_inputs[key] = value
                onnxscript_kwargs[key] = input
            else:
                onnxscript_kwargs[key] = value

        with onnxscript.evaluator.default_as(tracer):
            symbolic_outputs = function(*onnxscript_args, **onnxscript_kwargs)
        if not isinstance(symbolic_outputs, Sequence):
            symbolic_outputs = (symbolic_outputs,)

        # We need to set the size of the output tensors for the ONNX model to be valid
        for output, symbolic_output in zip(outputs, symbolic_outputs):
            if isinstance(output, Sequence):
                # Output is a sequence, set the type correctly to ListType
                symbolic_output.dtype = output[0].dtype
                symbolic_output.symbolic_value().setType(torch.ListType.ofTensors())
                continue
            output = (
                output
                if isinstance(output, torch.Tensor)
                else torch.tensor(output, device="cpu")
            )
            symbolic_output.shape = output.shape
            symbolic_output.dtype = output.dtype

        onnxscript_graph.register_outputs(symbolic_outputs)

        onnx_model = onnxscript_graph.to_model_proto(TEST_OPSET_VERSION)
        # Make sure the model is valid
        try:
            onnx.checker.check_model(onnx_model, full_check=True)
        except onnx.checker.ValidationError as e:
            raise AssertionError(
                f"ONNX model is invalid: {e}. "
                f"Model:\n"
                f"{onnxscript.proto2text(onnx_model)}"
            ) from e

        try:
            if os.environ.get("CATCH_ORT_SEGFAULT") == "1":
                # Use an individual process to run ONNX Runtime to catch segfaults
                return _safe_ort_session_run(onnx_model.SerializeToString(), ort_inputs)

            return _ort_session_run(onnx_model.SerializeToString(), ort_inputs)
        except (
            # pylint: disable=c-extension-no-member
            onnxruntime.capi.onnxruntime_pybind11_state.Fail,
            onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
            onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,
            # pylint: enable=c-extension-no-member
        ) as e:
            raise AssertionError(
                "ONNX Runtime failed to evaluate:\n"
                + _format_model_and_input_information(onnx_model, ort_inputs)
            ) from e
        except OrtAbortedError as e:
            raise AssertionError(
                "ONNX Runtime aborted:\n"
                + _format_model_and_input_information(onnx_model, ort_inputs)
            ) from e

    return _capture_graph_and_evaluate_torch_script_evaluator


def _eager_executor(
    outputs,
) -> Callable[[Callable[..., Any], tuple[Any], dict[str, Any]], None]:
    """Eagerly executes a function."""

    del outputs  # Unused

    def executor(function, args, kwargs):
        return function(*args, **kwargs)

    return executor


def run_test_output_match(
    test_suite: unittest.TestCase,
    device: str,
    dtype: torch.dtype,
    op: opinfo_core.OpInfo,
    function_executor: Callable,
):
    """Base test method for testing each opset, used by instantiate_device_type_tests.

    Args:
        test_suite: The test class instance.
        device: The PyTorch device. instantiate_device_type_tests provides this.
        dtype: The PyTorch dtype. instantiate_device_type_tests provides this.
        op: The OpInfo instance. instantiate_device_type_tests provides this.
        function_executor: The function executor. This is a function that takes
            a function and its arguments and returns the output of the function.
    """
    samples = op.sample_inputs(
        device,
        dtype,
        requires_grad=False,
    )

    onnx_function_and_wrangler = OPINFO_FUNCTION_MAPPING[op.name]
    input_wrangler = None
    if isinstance(onnx_function_and_wrangler, tuple):
        # Obtain the input_wrangler that manipulates the OpInfo inputs
        # to match the aten operator signature
        # An example is nn.functional.upsample_nearest2d, which has a different signature
        # than the aten operator upsample_nearest2d
        onnx_function, input_wrangler = onnx_function_and_wrangler
    else:
        assert callable(onnx_function_and_wrangler)
        onnx_function = onnx_function_and_wrangler

    for i, cpu_sample in enumerate(samples):
        inputs = (cpu_sample.input, *cpu_sample.args)
        # Provide the repr to subtest because tensors are not serializable in parallel test runs
        with test_suite.subTest(
            sample_num=i,
            inputs=repr(
                [
                    f"Tensor<{inp.shape}, dtype={inp.dtype}>"
                    if isinstance(inp, torch.Tensor)
                    else inp
                    for inp in inputs
                ]
            ),
            kwargs=repr(cpu_sample.kwargs),
        ):
            skip_reason = _should_skip_test_sample(op.name, cpu_sample)
            if skip_reason is not None:
                # Cannot use self.skip because pytest would skip the entire test
                warnings.warn(f"skipped sample {i}. Reason: {skip_reason}", stacklevel=1)
                continue
            input_onnx = [_convert_tensor_to_numpy(x) for x in inputs]
            kwargs_onnx = _convert_kwargs_for_onnx(cpu_sample.kwargs)
            if input_wrangler:
                input_onnx, kwargs_onnx = input_wrangler(input_onnx, kwargs_onnx)
            torch_output = op(*inputs, **cpu_sample.kwargs)

            reference_torch_outputs, _ = pytree.tree_flatten(torch_output)
            if op.name.startswith("split"):
                # Hack for handling split
                # Split returns a Sequence that should be treats as a single
                # value. So we wrap it into a tuple.
                # TODO(justinchuby): Find a more general solution
                reference_torch_outputs = [reference_torch_outputs]

            function_output = function_executor(reference_torch_outputs)(
                onnx_function, input_onnx, kwargs_onnx
            )
            # Finally we re-flatten everything
            # TODO: add pytree structure comparison.
            flattened_torch_outputs, _ = pytree.tree_flatten(torch_output)
            flattened_function_outputs, _ = pytree.tree_flatten(function_output)

            assert flattened_torch_outputs
            assert len(flattened_torch_outputs) == len(flattened_function_outputs)

            for j, (torch_output, function_output) in enumerate(
                zip(flattened_torch_outputs, flattened_function_outputs)
            ):
                if dtype == torch.float32:
                    # Relax atol and rtol for float32 based on empirical results
                    # The current most relaxed values are for aten::matmul
                    rtol = 3.7e-5
                    atol = 1.8e-4
                else:
                    rtol = None
                    atol = None

                if not isinstance(function_output, np.ndarray):
                    # An onnxscript tensor
                    function_output = function_output.value

                actual = torch.tensor(function_output)
                expected = (
                    torch_output
                    if isinstance(torch_output, torch.Tensor)
                    else torch.tensor(torch_output)
                )

                if op.name in NONDETERMINISTIC_OPS:
                    # Check shape and dtype only for ops that are known to be
                    # nondeterministic
                    test_suite.assertEqual(actual.shape, expected.shape)
                    test_suite.assertEqual(actual.dtype, expected.dtype)
                    continue

                # Use torch.testing as opposed to np.testing to ensure dtypes and shapes match
                try:
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=rtol,
                        atol=atol,
                        check_device=False,
                    )
                except AssertionError as e:
                    if len(flattened_torch_outputs) > 1:
                        raise AssertionError(f"Output {j} mismatch") from e
                    raise


class TestOutputConsistencyEager(unittest.TestCase):
    """Test output consistency between the ONNX op run with ONNX eager mode and PyTorch eager mode.

    This is a parameterized test suite.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        ort.set_seed(42)

    @add_decorate_info(
        OPS_DB,
        "TestOutputConsistencyEager",
        "test_output_match_opinfo_",
        skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
    )
    @common_device_type.ops(  # type: ignore[misc]
        [info for info in OPS_DB if info.name in TESTED_OPS],
        allowed_dtypes=TESTED_DTYPES,
    )
    def test_output_match_opinfo_(
        self, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo
    ):
        """Base test method for testing each op with the eager executor, used by instantiate_device_type_tests."""
        run_test_output_match(self, device, dtype, op, _eager_executor)


class TestOutputConsistencyFullGraph(unittest.TestCase):
    """Test output consistency between exported ONNX op run as a graph and PyTorch eager mode.

    This is a parameterized test suite.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        ort.set_seed(42)

    @add_decorate_info(
        OPS_DB,
        "TestOutputConsistencyFullGraph",
        "test_output_match_opinfo_",
        skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
    )
    @common_device_type.ops(  # type: ignore[misc]
        [info for info in OPS_DB if info.name in TESTED_OPS],
        allowed_dtypes=TESTED_DTYPES,
    )
    def test_output_match_opinfo_(
        self, device: str, dtype: torch.dtype, op: opinfo_core.OpInfo
    ):
        """Base test method for testing each op by running the full ONNX graph."""
        run_test_output_match(self, device, dtype, op, _graph_executor)


common_device_type.instantiate_device_type_tests(
    TestOutputConsistencyEager, globals(), only_for="cpu"
)

common_device_type.instantiate_device_type_tests(
    TestOutputConsistencyFullGraph, globals(), only_for="cpu"
)

if __name__ == "__main__":
    unittest.main()
