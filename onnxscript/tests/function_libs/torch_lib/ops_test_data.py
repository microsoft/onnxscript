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
import dataclasses
import functools
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo import definitions as opinfo_definitions

import onnxscript
import onnxscript.evaluator
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


@dataclasses.dataclass
class TorchLibOpInfo:
    """A dataclass to store the information to test an torchlib op."""

    # The name of the op_info, e.g. "add"
    op_info_name: str
    # The torchlib ONNX Function to test
    op: Callable[..., Any]
    # Explicitly specify when the op is trace_only
    trace_only: bool = False
    # The input wrangler function to adjust the input to fit the aten signature
    input_wrangler: Optional[
        Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]]
    ] = None
    # Whether the op is non-deterministic
    nondeterministic: bool = False
    # Whether the function is designed for complex inputs
    complex: bool = False
    # Expected skips or fails for the test and/or subtests
    skips_or_fails: tuple[ops_test_common.DecorateMeta, ...] = ()


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


def _mean_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Make the dims as tensor
    if "dim" in kwargs:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
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


def _scatter_reduce_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Put the string into kwargs, otherwise FullGraph mode could not find get 'reduce' argument
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
TESTED_TORCHLIB_OPS: tuple[TorchLibOpInfo, ...] = (
    TorchLibOpInfo(
        "all_dim",
        core_ops.aten_all_dim,
        skips_or_fails=(
            xfail(
                "all_dim",
                matcher=lambda sample: not (len(sample.kwargs) > 0),
                reason="this Aten overload only support one tensor as input and {dim,keepdim} as kwargs by design",
            ),
        ),
    ),
    TorchLibOpInfo("allclose", core_ops.aten_allclose),
    TorchLibOpInfo(
        "all",
        core_ops.aten_all,
        skips_or_fails=(
            skip(
                "all",
                matcher=lambda sample: not (len(sample.kwargs) == 0),
                reason="this Aten overload only support one tensor as input by design",
            ),
        ),
    ),
    TorchLibOpInfo("abs", core_ops.aten_abs),
    TorchLibOpInfo("abs", core_ops.aten_abs_complex, complex=True),
    TorchLibOpInfo("acos", core_ops.aten_acos),
    TorchLibOpInfo("acosh", core_ops.aten_acosh),
    TorchLibOpInfo("add", core_ops.aten_add),
    TorchLibOpInfo("addmm", core_ops.aten_addmm),
    TorchLibOpInfo(
        "amax",
        core_ops.aten_amax,
        input_wrangler=_amin_amax_input_wrangler,
        skips_or_fails=(
            skip(
                "amax",
                matcher=lambda sample: len(sample.input.shape) == 0,
                reason="fixme (core dump): ORT aborts on scalar inputs to ReduceMax-18",
            ),
        ),
    ),
    TorchLibOpInfo(
        "amin",
        core_ops.aten_amin,
        input_wrangler=_amin_amax_input_wrangler,
        skips_or_fails=(
            skip(
                "amin",
                matcher=lambda sample: len(sample.input.shape) == 0,
                reason="fixme (core dump): ORT aborts on scalar inputs to ReduceMin-18",
            ),
        ),
    ),
    TorchLibOpInfo(
        "any",
        core_ops.aten_any,
        skips_or_fails=(
            skip(
                "any",
                matcher=lambda sample: not (len(sample.kwargs) == 0),
                reason="this Aten overload only support one tensor as input by design",
            ),
        ),
    ),
    TorchLibOpInfo("any_dim", core_ops.aten_any_dim),
    TorchLibOpInfo("asin", core_ops.aten_asin),
    TorchLibOpInfo("asinh", core_ops.aten_asinh),
    TorchLibOpInfo("atan", core_ops.aten_atan),
    TorchLibOpInfo("atan2", core_ops.aten_atan2),
    TorchLibOpInfo("atanh", core_ops.aten_atanh),
    TorchLibOpInfo("atleast_1d", core_ops.aten_atleast_1d),
    TorchLibOpInfo(
        "atleast_1d_single_tensor",
        core_ops.aten_atleast_1d_single_tensor,
        skips_or_fails=(
            skip(
                "atleast_1d_single_tensor",
                matcher=lambda sample: isinstance(sample.input, (list, tuple)),
                reason="atleast_1d_single_tensor overload takes single tensor as input",
            ),
        ),
    ),
    TorchLibOpInfo("atleast_2d", core_ops.aten_atleast_2d),
    TorchLibOpInfo(
        "atleast_2d_single_tensor",
        core_ops.aten_atleast_2d_single_tensor,
        skips_or_fails=(
            skip(
                "atleast_2d_single_tensor",
                matcher=lambda sample: isinstance(sample.input, (list, tuple)),
                reason="atleast_2d_single_tensor overload takes single tensor as input",
            ),
        ),
    ),
    TorchLibOpInfo("atleast_3d", core_ops.aten_atleast_3d),
    TorchLibOpInfo(
        "atleast_3d_single_tensor",
        core_ops.aten_atleast_3d_single_tensor,
        skips_or_fails=(
            skip(
                "atleast_3d_single_tensor",
                matcher=lambda sample: isinstance(sample.input, (list, tuple)),
                reason="atleast_3d_single_tensor overload takes single tensor as input",
            ),
        ),
    ),
    TorchLibOpInfo("baddbmm", core_ops.aten_baddbmm),
    TorchLibOpInfo("bmm", core_ops.aten_bmm),
    TorchLibOpInfo("broadcast_to", core_ops.aten_broadcast_to),
    TorchLibOpInfo(
        "cat",
        (core_ops.aten_cat),
        input_wrangler=_cat_input_wrangler,
        skips_or_fails=(
            skip(
                "cat",
                matcher=lambda sample: sample.input[0].equal(torch.tensor([])),
                reason="cat does not support zero-dim tensors yet",
            ),
        ),
    ),
    TorchLibOpInfo("ceil", core_ops.aten_ceil),
    TorchLibOpInfo("chunk", core_ops.aten_chunk),
    TorchLibOpInfo("clamp_max", core_ops.aten_clamp_max),
    TorchLibOpInfo("clamp_min", core_ops.aten_clamp_min),
    TorchLibOpInfo("clone", core_ops.aten_clone),
    TorchLibOpInfo("constant_pad_nd", core_ops.aten_constant_pad_nd),
    # TorchLibOpInfo("copy", core_ops.aten_copy),  # copy is not in OPS_DB
    TorchLibOpInfo("cos", core_ops.aten_cos),
    TorchLibOpInfo("cosh", core_ops.aten_cosh),
    TorchLibOpInfo("cross", core_ops.aten_cross),
    # TorchLibOpInfo("detach", core_ops.aten_detach),  # detach is not in OP-TEST-DB
    TorchLibOpInfo(
        "div",
        core_ops.aten_div,
        skips_or_fails=(
            skip(
                "div",
                matcher=lambda sample: sample.kwargs.get("rounding_mode") is not None,
                reason="rounding_mode is not yet supported",
            ),
        ),
    ),
    TorchLibOpInfo("dot", core_ops.aten_dot),
    TorchLibOpInfo(
        "empty",
        (core_ops.aten_empty),
        input_wrangler=_empty_input_wrangler,
        nondeterministic=True,
    ),
    # TorchLibOpInfo("empty_strided", core_ops.aten_empty_strided),  # empty_strided is not in OPS_DB
    TorchLibOpInfo("eq", core_ops.aten_eq),
    TorchLibOpInfo("equal", core_ops.aten_equal),
    TorchLibOpInfo("exp", core_ops.aten_exp),
    TorchLibOpInfo("exp2", core_ops.aten_exp2),
    TorchLibOpInfo("expand", core_ops.aten_expand),
    TorchLibOpInfo("expand_as", core_ops.aten_expand_as),
    TorchLibOpInfo("erf", core_ops.aten_erf),
    TorchLibOpInfo("fill", core_ops.aten_fill),
    TorchLibOpInfo("flip", (core_ops.aten_flip), input_wrangler=_flip_input_wrangler),
    TorchLibOpInfo("floor", core_ops.aten_floor),
    TorchLibOpInfo("fmod", core_ops.aten_fmod),
    TorchLibOpInfo("full", core_ops.aten_full),
    TorchLibOpInfo(
        "full_like_dtype",
        core_ops.aten_full_like_dtype,
        skips_or_fails=(
            skip(
                "full_like_dtype",
                matcher=lambda sample: "dtype" not in sample.kwargs,
                reason="this Aten overload only support dtype in kwargs",
            ),
        ),
    ),
    TorchLibOpInfo(
        "full_like",
        core_ops.aten_full_like,
        skips_or_fails=(
            skip(
                "full_like",
                matcher=lambda sample: ("dtype" in sample.kwargs),
                reason="this Aten overload only support dtype not in kwargs",
            ),
        ),
    ),
    TorchLibOpInfo("gather", core_ops.aten_gather),
    TorchLibOpInfo("ge", core_ops.aten_ge),
    # TorchLibOpInfo("greater_equal", core_ops.aten_greater_equal),  # no test case in OPS_DB
    # TorchLibOpInfo("greater", core_ops.aten_greater),  # no test case in OPS_DB
    TorchLibOpInfo("gt", core_ops.aten_gt),
    # TorchLibOpInfo("is_same_size", core_ops.aten_is_same_size),  # no test case in OPS_DB
    # TorchLibOpInfo("is_nonzero", core_ops.aten_is_nonzero),  # no test case in OPS_DB
    TorchLibOpInfo(
        "index_put_bool",
        core_ops.aten_index_put_bool,
        skips_or_fails=(
            skip(
                "index_put_bool",
                matcher=lambda sample: not (sample.args[0][0].dtype == torch.bool),
                reason="this Aten overload only support tensor(bool) as args",
            ),
        ),
    ),
    TorchLibOpInfo(
        "index_put",
        core_ops.aten_index_put,
        skips_or_fails=(
            skip(
                "index_put",
                matcher=lambda sample: not (sample.args[0][0].dtype == torch.int64),
                reason="this Aten overload only support tensor(int) as args",
            ),
        ),
    ),
    TorchLibOpInfo("index_select", core_ops.aten_index_select),
    TorchLibOpInfo("isclose", core_ops.aten_isclose),
    TorchLibOpInfo("isfinite", core_ops.aten_isfinite),
    TorchLibOpInfo("isinf", core_ops.aten_isinf),
    TorchLibOpInfo("isnan", core_ops.aten_isnan),
    TorchLibOpInfo("isneginf", core_ops.aten_isneginf),
    TorchLibOpInfo("isposinf", core_ops.aten_isposinf),
    TorchLibOpInfo("log", core_ops.aten_log),
    TorchLibOpInfo("le", core_ops.aten_le),
    TorchLibOpInfo("log10", core_ops.aten_log10),
    TorchLibOpInfo("log1p", core_ops.aten_log1p),
    TorchLibOpInfo("log_softmax", special_ops.aten_special_log_softmax),
    TorchLibOpInfo("log2", core_ops.aten_log2),
    TorchLibOpInfo("logaddexp", core_ops.aten_logaddexp),
    TorchLibOpInfo("logaddexp2", core_ops.aten_logaddexp2),
    TorchLibOpInfo("logcumsumexp", core_ops.aten_logcumsumexp),
    TorchLibOpInfo("logdet", core_ops.aten_logdet),
    TorchLibOpInfo(
        "logsumexp",
        core_ops.aten_logsumexp,
        skips_or_fails=(
            xfail("logcumsumexp", reason="fixme: naive implementation not numerically stable"),
        ),
    ),
    TorchLibOpInfo("lt", core_ops.aten_lt),
    TorchLibOpInfo("masked_fill", core_ops.aten_masked_fill),
    TorchLibOpInfo(
        "matmul",
        core_ops.aten_matmul,
        skips_or_fails=(
            skip(
                "matmul",
                matcher=lambda sample: torch.numel(sample.input) == 0,
                reason="values of matmul of [m, 0] and [0, n] matrices are undefined",
            ),
        ),
    ),
    TorchLibOpInfo("maximum", core_ops.aten_maximum),
    TorchLibOpInfo(
        "mean",
        (core_ops.aten_mean),
        input_wrangler=_mean_input_wrangler,
        skips_or_fails=(
            skip(
                "mean",
                matcher=lambda sample: sample.kwargs.get("dim") is not None,
                reason="this Aten overload only accept 1 inputs: self",
            ),
        ),
    ),
    TorchLibOpInfo(
        "mean_dim",
        (core_ops.aten_mean_dim),
        input_wrangler=_mean_input_wrangler,
        skips_or_fails=(
            skip(
                "mean_dim",
                matcher=lambda sample: sample.kwargs.get("dim") is None,
                reason="this Aten overload can accept 2 inputs:(self, dim)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "min_dim",
        core_ops.aten_min_dim,
        skips_or_fails=(
            xfail(
                "min_dim",
                variant_name="reduction_with_dim",
                reason="ORT Graph attribute inferencing failed https://github.com/onnx/onnx/issues/4986",
                test_class_name="TestOutputConsistencyFullGraph",
            ),
            xfail(
                "min_dim",  # aten_min_dim(self, dim)
                matcher=lambda sample: len(sample.args) == 0
                or (len(sample.args) > 0 and not isinstance(sample.args[0], int)),
                reason="this ATen overload only support one tensor as input and another int as args",
            ),
        ),
    ),
    TorchLibOpInfo(
        "min_other",
        core_ops.aten_min_other,
        skips_or_fails=(
            xfail(
                "min_other",  # aten_min_other(self, other)
                matcher=lambda sample: len(sample.args) == 0
                or (len(sample.args) > 0 and isinstance(sample.args[0], int)),
                reason="this ATen overload only support one tensor as input and another tensor as args",
            ),
        ),
    ),
    TorchLibOpInfo(
        "min",
        core_ops.aten_min,
        skips_or_fails=(
            skip(
                "min",  # aten_min
                matcher=lambda sample: len(sample.args) > 0,
                reason="this ATen overload only supports one tensor as input by design",
            ),
        ),
    ),
    TorchLibOpInfo("minimum", core_ops.aten_minimum),
    TorchLibOpInfo("mm", core_ops.aten_mm),
    TorchLibOpInfo("mul", core_ops.aten_mul),
    TorchLibOpInfo("narrow", core_ops.aten_narrow),
    # TorchLibOpInfo("native_dropout", core_ops.aten_native_dropout),  # native_dropout is not in OPS_DB
    TorchLibOpInfo("ne", core_ops.aten_ne),
    TorchLibOpInfo("neg", core_ops.aten_neg),
    TorchLibOpInfo(
        "new_empty_dtype",
        core_ops.aten_new_empty_dtype,
        nondeterministic=True,
        skips_or_fails=(
            skip(
                "new_empty_dtype",
                matcher=lambda sample: sample.kwargs.get("dtype") is None,
                reason="this Aten overload must have 3 inputs:(self, size, dtype)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_empty",
        core_ops.aten_new_empty,
        nondeterministic=True,
        skips_or_fails=(
            skip(
                "new_empty",
                matcher=lambda sample: sample.kwargs.get("dtype") is not None,
                reason="this Aten overload only accept 2 inputs:(self, size)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_empty_strided_dtype",
        core_ops.aten_new_empty_strided_dtype,
        nondeterministic=True,
        skips_or_fails=(
            skip(
                "new_empty_strided_dtype",
                matcher=lambda sample: sample.kwargs.get("dtype") is None,
                reason="this Aten overload must have 4 inputs:(self, size, stride, dtype)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_empty_strided",
        core_ops.aten_new_empty_strided,
        nondeterministic=True,
        skips_or_fails=(
            skip(
                "new_empty_strided",
                matcher=lambda sample: sample.kwargs.get("dtype") is not None,
                reason="this Aten overload only accept 3 inputs:(self, size, stride)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_full_dtype",
        core_ops.aten_new_full_dtype,
        skips_or_fails=(
            skip(
                "new_full_dtype",
                matcher=lambda sample: sample.kwargs.get("dtype") is None,
                reason="this Aten overload must have 4 inputs:(self, size, fill_value, dtype)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_full",
        core_ops.aten_new_full,
        skips_or_fails=(
            skip(
                "new_full",
                matcher=lambda sample: sample.kwargs.get("dtype") is not None,
                reason="this Aten overload only accept 3 inputs:(self, size, fill_value)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_ones_dtype",
        core_ops.aten_new_ones_dtype,
        skips_or_fails=(
            skip(
                "new_ones_dtype",
                matcher=lambda sample: sample.kwargs.get("dtype") is None,
                reason="",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_ones",
        core_ops.aten_new_ones,
        skips_or_fails=(
            skip(
                "new_ones",
                matcher=lambda sample: sample.kwargs.get("dtype") is not None,
                reason="",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_zeros_dtype",
        core_ops.aten_new_zeros_dtype,
        skips_or_fails=(
            skip(
                "new_zeros_dtype",
                matcher=lambda sample: sample.kwargs.get("dtype") is None,
                reason="",
            ),
        ),
    ),
    TorchLibOpInfo(
        "new_zeros",
        core_ops.aten_new_zeros,
        skips_or_fails=(
            skip(
                "new_zeros",
                matcher=lambda sample: sample.kwargs.get("dtype") is not None,
                reason="",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.adaptive_avg_pool1d",
        nn_ops.aten_adaptive_avg_pool1d,
        skips_or_fails=(
            xfail(
                "nn.functional.adaptive_avg_pool1d",
                # Shape should be [N, C, D1]
                matcher=lambda sample: sample.args[0] not in {1, (1,)},
                reason="only global pooling is supported; only batched inputs are supported",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.adaptive_avg_pool2d",
        nn_ops.aten_adaptive_avg_pool2d,
        skips_or_fails=(
            xfail(
                "nn.functional.adaptive_avg_pool2d",
                matcher=lambda sample: sample.args[0] != (1, 1),
                reason="only global pooling is supported; only batched inputs are supported",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.adaptive_avg_pool3d",
        nn_ops.aten_adaptive_avg_pool3d,
        skips_or_fails=(
            xfail(
                "nn.functional.adaptive_avg_pool3d",
                matcher=lambda sample: sample.args[0] != (1, 1, 1),
                reason="only global pooling is supported; only batched inputs are supported",
            ),
        ),
    ),
    TorchLibOpInfo("nn.functional.celu", nn_ops.aten_celu),
    TorchLibOpInfo(
        "nn.functional.cross_entropy",
        # use cross_entropy as test case instead of cross_entropy_loss (not in OPS_DB)
        (nn_ops.aten_cross_entropy_loss),
        input_wrangler=_cross_entropy_input_wrangler,
        skips_or_fails=(
            xfail(
                "nn.functional.cross_entropy",
                matcher=lambda sample: len(sample.args) < 1
                or (
                    isinstance(sample.args[0], torch.Tensor)
                    and sample.args[0].dtype != torch.int64
                ),
                reason="ONNX SoftmaxCrossEntropyLoss op only accept argument[target] as int type",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.dropout",
        (core_ops.aten_dropout),
        input_wrangler=_dropout_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.dropout",
                matcher=lambda sample: len(sample.kwargs) == 0
                or sample.kwargs.get("p", 0.0) > 0.0,
                reason="dropout is random so the result not match",
            ),
        ),
    ),
    TorchLibOpInfo("nn.functional.elu", nn_ops.aten_elu),
    TorchLibOpInfo(
        "nn.functional.embedding",
        (core_ops.aten_embedding),
        input_wrangler=_embedding_input_wrangler,
    ),
    TorchLibOpInfo("nn.functional.hardtanh", nn_ops.aten_hardtanh),
    TorchLibOpInfo("nn.functional.leaky_relu", nn_ops.aten_leaky_relu),
    TorchLibOpInfo(
        "nn.functional.logsigmoid",
        nn_ops.aten_log_sigmoid,
        skips_or_fails=(
            xfail(
                "nn.functional.logsigmoid",
                dtypes=[torch.float16],
                reason="Eager mode failed on case(0,2) at location(0,6) due to precision loss",
                test_class_name="TestOutputConsistencyEager",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.nll_loss_weight",
        (nn_ops.aten_nll_loss_weight),
        input_wrangler=_nll_loss_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.nll_loss_weight",
                matcher=lambda sample: "weight" not in sample.kwargs,
                reason="this Aten overload need weight as kwargs",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.nll_loss",
        (nn_ops.aten_nll_loss),
        input_wrangler=_nll_loss_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.nll_loss",
                matcher=lambda sample: "weight" in sample.kwargs,
                reason="this Aten overload doesn't accept weight as kwargs",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.reflection_pad2d",
        (nn_ops.aten_reflection_pad2d),
        input_wrangler=_reflection_pad2d_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.reflection_pad2d",
                matcher=lambda sample: not (
                    len(sample.args) > 1 and sample.args[1] == "reflect"
                ),
                reason="this Aten overload need args[1] == 'reflect' for pad mode",
            ),
        ),
    ),
    TorchLibOpInfo("nn.functional.relu", nn_ops.aten_relu),
    TorchLibOpInfo("nn.functional.relu6", nn_ops.aten_relu6),
    TorchLibOpInfo(
        "nn.functional.replication_pad2d",
        (nn_ops.aten_replication_pad2d),
        input_wrangler=_replication_pad2d_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.replication_pad2d",
                matcher=lambda sample: not (
                    len(sample.args) > 1 and sample.args[1] == "replicate"
                ),
                reason="this Aten overload need args[1] == 'replicate' for pad mode",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.replication_pad3d",
        (nn_ops.aten_replication_pad3d),
        input_wrangler=_replication_pad3d_input_wrangler,
        skips_or_fails=(
            skip(
                "nn.functional.replication_pad3d",
                matcher=lambda sample: not (
                    len(sample.args) > 1
                    and sample.args[1] == "replicate"
                    and len(sample.input.shape) == 5
                ),
                reason="this Aten overload need args[1] == 'replicate' for pad mode, and 3D tensor",
            ),
        ),
    ),
    TorchLibOpInfo("nn.functional.selu", core_ops.aten_selu),
    TorchLibOpInfo(
        "nn.functional.mse_loss",
        (nn_ops.aten_mse_loss),
        input_wrangler=_mse_loss_input_wrangler,
    ),
    TorchLibOpInfo(
        "nonzero",
        core_ops.aten_nonzero,
        skips_or_fails=(
            xfail(
                "nonzero",
                matcher=lambda sample: sample.kwargs.get("as_tuple") is not None,
                reason="as_tuple=True is not supported",
            ),
        ),
    ),
    TorchLibOpInfo(
        "normal",
        core_ops.aten_normal,
        nondeterministic=True,
        skips_or_fails=(
            skip(
                "normal",
                matcher=lambda sample: len(sample.args) > 0
                and not isinstance(sample.args[0], float),
                reason="ORT only accept float type for args[0] 'mean'",
            ),
        ),
    ),
    TorchLibOpInfo("ones", core_ops.aten_ones),
    TorchLibOpInfo(
        "permute",
        (core_ops.aten_permute),
        input_wrangler=_permute_input_wrangler,
        skips_or_fails=(
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
        ),
    ),
    TorchLibOpInfo("pow", core_ops.aten_pow),
    # TorchLibOpInfo("rand", core_ops.aten_rand),  # no test case in OPS_DB
    TorchLibOpInfo(
        "randn",
        (core_ops.aten_randn),
        input_wrangler=_randn_input_wrangler,
        nondeterministic=True,
    ),
    TorchLibOpInfo("reciprocal", core_ops.aten_reciprocal),
    TorchLibOpInfo(
        "remainder",
        core_ops.aten_remainder,
        skips_or_fails=(
            xfail(
                "remainder",
                dtypes=[torch.float16],
                reason="Eager mode failed on case(self=7.75,other=0.1582) due to precision loss",
                test_class_name="TestOutputConsistencyEager",
            ),
        ),
    ),
    TorchLibOpInfo("repeat", core_ops.aten_repeat),
    TorchLibOpInfo("reshape", core_ops.aten_reshape),
    TorchLibOpInfo("resolve_conj", core_ops.aten_resolve_conj),
    TorchLibOpInfo("resolve_neg", core_ops.aten_resolve_neg),
    TorchLibOpInfo(
        "round",
        core_ops.aten_round,
        skips_or_fails=(
            xfail(
                "round",
                variant_name="decimals_0",
                reason="The op does not support decimals yet",
                test_class_name="TestOutputConsistencyEager",
            ),
            xfail(
                "round",
                variant_name="decimals_3",
                reason="The op does not support decimals yet",
            ),
            xfail(
                "round",
                variant_name="decimals_neg_3",
                reason="The op does not support decimals yet",
            ),
        ),
    ),
    TorchLibOpInfo("rsqrt", core_ops.aten_rsqrt),
    TorchLibOpInfo("rsub", core_ops.aten_rsub),
    # TorchLibOpInfo("scalar_tensor", core_ops.aten_scalar_tensor),  # no test case in OPS_DB
    TorchLibOpInfo(
        "scatter_add",
        core_ops.aten_scatter_add,
        skips_or_fails=(
            xfail(
                "scatter_add",
                matcher=lambda sample: len(sample.input.shape) == 0,
                reason="fixme: Rank(0) input will lead ORT failed due to different rank(result) in if-else branch",
            ),
        ),
    ),
    TorchLibOpInfo("select", core_ops.aten_select),
    TorchLibOpInfo("sigmoid", core_ops.aten_sigmoid),
    TorchLibOpInfo("sign", core_ops.aten_sign),
    TorchLibOpInfo("sin", core_ops.aten_sin),
    TorchLibOpInfo("sinh", core_ops.aten_sinh),
    TorchLibOpInfo("softmax", special_ops.aten_special_softmax),
    TorchLibOpInfo("split_with_sizes", core_ops.aten_split_with_sizes),
    TorchLibOpInfo("split", core_ops.aten_split),
    TorchLibOpInfo("sqrt", core_ops.aten_sqrt),
    TorchLibOpInfo(
        "squeeze_dim",
        core_ops.aten_squeeze_dim,
        skips_or_fails=(
            skip(
                "squeeze_dim",
                matcher=lambda sample: not (
                    len(sample.args) > 0 and isinstance(sample.args[0], int)
                ),
                reason="this Aten overload only support one tensor as input and one int as args by design",
            ),
        ),
    ),
    TorchLibOpInfo(
        "squeeze",
        core_ops.aten_squeeze,
        skips_or_fails=(
            skip(
                "squeeze",
                matcher=lambda sample: not (len(sample.args) == 0),
                reason="this Aten overload only support one tensor as input by design",
            ),
        ),
    ),
    TorchLibOpInfo("stack", core_ops.aten_stack),
    TorchLibOpInfo("sub", core_ops.aten_sub),
    # TorchLibOpInfo("sym_size", core_ops.aten_sym_size),  # no test case in OPS_DB
    TorchLibOpInfo(
        "t",
        core_ops.aten_t,
        skips_or_fails=(
            xfail(
                "t",
                reason="ORT Graph attribute inferencing failed on rank-1 input",
                test_class_name="TestOutputConsistencyFullGraph",
            ),
        ),
    ),
    TorchLibOpInfo("tan", core_ops.aten_tan),
    TorchLibOpInfo("tanh", core_ops.aten_tanh),
    TorchLibOpInfo(
        "tile",
        core_ops.aten_tile,
        skips_or_fails=(
            skip(
                "tile",
                matcher=lambda sample: any(dim == 0 for dim in sample.input.shape)
                or not sample.input.shape,
                reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
            ),
        ),
    ),
    TorchLibOpInfo("topk", core_ops.aten_topk),
    TorchLibOpInfo("tril", core_ops.aten_tril),
    TorchLibOpInfo("triu", core_ops.aten_triu),
    TorchLibOpInfo("trunc", core_ops.aten_trunc),
    TorchLibOpInfo(
        "unflatten",
        (core_ops.aten_unflatten),
        input_wrangler=_unflatten_input_wrangler,
        skips_or_fails=(
            xfail(
                "unflatten",
                reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
                test_class_name="TestOutputConsistencyFullGraph",
            ),
            xfail(
                "unflatten",
                matcher=lambda sample: any(dim == 0 for dim in sample.input.shape),
                reason="fixme: Logic not implemented for size 0 inputs in op.Reshape",
            ),
        ),
    ),
    TorchLibOpInfo("unsqueeze", core_ops.aten_unsqueeze),
    TorchLibOpInfo("view", core_ops.aten_view),
    TorchLibOpInfo(
        "vstack",
        core_ops.aten_vstack,
        skips_or_fails=(
            xfail(
                "vstack",
                reason="fixme: A bug of constant-propagation optimization within the subgraph, we can avoid it by turning off graph-optimizations in session options",
            ),
        ),
    ),
    TorchLibOpInfo("where", (core_ops.aten_where), input_wrangler=_where_input_wrangler),
    TorchLibOpInfo("xlogy", special_ops.aten_special_xlogy),
    TorchLibOpInfo("zeros", core_ops.aten_zeros),
    TorchLibOpInfo(
        "arange_start_step",
        core_ops.aten_arange_start_step,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "arange_start_step",
                matcher=lambda sample: len(sample.args) != 2,
                reason="arange_start_step overload takes three arguments (input, start, step)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "arange_start",
        core_ops.aten_arange_start,
        trace_only=True,
        skips_or_fails=(
            skip(
                "arange_start",
                matcher=lambda sample: len(sample.args) != 1,
                reason="arange_start overload takes two arguments (input, start)",
            ),
        ),
    ),
    TorchLibOpInfo(
        "arange",
        core_ops.aten_arange,
        trace_only=True,
        skips_or_fails=(
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
        ),
    ),
    TorchLibOpInfo("argmax", core_ops.aten_argmax, trace_only=True),
    TorchLibOpInfo("argmin", core_ops.aten_argmin, trace_only=True),
    TorchLibOpInfo(
        "as_strided",
        core_ops.aten_as_strided,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "as_strided",
                variant_name="partial_views",
                reason="ONNX doesn't have partial view for tensor",
            ),
        ),
    ),
    TorchLibOpInfo("clamp", core_ops.aten_clamp, trace_only=True),
    TorchLibOpInfo("col2im", nn_ops.aten_col2im, trace_only=True),
    TorchLibOpInfo("cumsum", core_ops.aten_cumsum, trace_only=True),
    TorchLibOpInfo("contiguous", core_ops.aten_contiguous, trace_only=True),
    TorchLibOpInfo("convolution", core_ops.aten_convolution, trace_only=True),
    TorchLibOpInfo(
        "empty_like", core_ops.aten_empty_like, nondeterministic=True, trace_only=True
    ),
    TorchLibOpInfo(
        "grid_sampler_2d",
        core_ops.aten_grid_sampler_2d,
        trace_only=True,
        skips_or_fails=(
            skip(
                "grid_sampler_2d",
                # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
                matcher=lambda sample: sample.args[1] == 2,
                reason="fixme: 'bicubic' mode in ORT implemented differently with Torch",
            ),
        ),
    ),
    TorchLibOpInfo(
        "hstack",
        core_ops.aten_hstack,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "hstack",
                reason="fixme: A bug of constant-propagation optimization within the subgraph, we can avoid it by turning off graph-optimizations in session options",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.grid_sample",
        (core_ops.aten_grid_sampler),
        input_wrangler=_grid_sample_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.grid_sample",
                # Torch implemented this using the cubic convolution algorithm with alhpa=-0.75, might be different than ORT
                matcher=lambda sample: sample.kwargs.get("mode") == "bicubic"
                or len(sample.args[0].shape) != 4,
                reason="fixme: 'bicubic' mode in ORT implemented differently with Torch and only support 4D-tensor",
            ),
        ),
    ),
    TorchLibOpInfo("layer_norm", core_ops.aten_layer_norm, trace_only=True),
    TorchLibOpInfo("logit", core_ops.aten_logit, trace_only=True),
    TorchLibOpInfo(
        "max",
        core_ops.aten_max,
        trace_only=True,
        skips_or_fails=(
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
        ),
    ),
    TorchLibOpInfo(
        "max_pool1d", nn_ops.aten_max_pool1d, trace_only=True
    ),  # Custom from extra_opinfo
    TorchLibOpInfo(
        "max_pool2d", nn_ops.aten_max_pool2d, trace_only=True
    ),  # Custom from extra_opinfo
    TorchLibOpInfo(
        "max_pool3d",
        nn_ops.aten_max_pool3d,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "max_pool3d",
                variant_name="empty_strides",
                reason="fixme: 'shape' do not match: torch.Size([2, 3, 4, 3]) != torch.Size([2, 3, 4, 2])",
            ),
        ),
    ),  # Custom from extra_opinfo
    TorchLibOpInfo("native_batch_norm", core_ops.aten_native_batch_norm, trace_only=True),
    TorchLibOpInfo("native_group_norm", core_ops.aten_native_group_norm, trace_only=True),
    TorchLibOpInfo("native_layer_norm", core_ops.aten_native_layer_norm, trace_only=True),
    TorchLibOpInfo(
        "nn.functional.avg_pool2d",
        (nn_ops.aten_avg_pool2d),
        input_wrangler=_avg_pool2d_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "nn.functional.avg_pool2d",
                matcher=lambda sample: len(sample.args) > 5 and sample.args[5] is not None,
                reason="ONNX doesn't support divisor_override argument",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.conv1d",
        core_ops.aten_conv1d,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "nn.functional.conv1d",
                matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
                reason="String padding is not accepted by aten::conv1d",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.conv2d",
        core_ops.aten_conv2d,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "nn.functional.conv2d",
                matcher=lambda sample: isinstance(sample.kwargs.get("padding"), str),
                reason="String padding is not accepted by aten::conv2d",
            ),
        ),
    ),
    TorchLibOpInfo("nn.functional.conv3d", core_ops.aten_conv3d, trace_only=True),
    TorchLibOpInfo("nn.functional.gelu", nn_ops.aten_gelu, trace_only=True),
    TorchLibOpInfo("nn.functional.linear", nn_ops.aten_linear, trace_only=True),
    TorchLibOpInfo(
        "nn.functional.max_pool1d",
        (nn_ops.aten_max_pool1d),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.max_pool1d",
                matcher=lambda sample: sample.kwargs.get("return_indices") is True,
                reason="this aten overload assume return_indices=False",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool1d_with_indices",
        (nn_ops.aten_max_pool1d_with_indices),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.max_pool1d_with_indices",
                matcher=lambda sample: sample.kwargs.get("return_indices") is False,
                reason="this aten overload assume return_indices=True",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool2d",
        (nn_ops.aten_max_pool2d),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.max_pool2d",
                matcher=lambda sample: sample.kwargs.get("return_indices") is True,
                reason="this aten overload assume return_indices=False",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool2d_with_indices",
        (nn_ops.aten_max_pool2d_with_indices),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.max_pool2d_with_indices",
                matcher=lambda sample: sample.kwargs.get("return_indices") is False,
                reason="this aten overload assume return_indices=True",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool3d",
        (nn_ops.aten_max_pool3d),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
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
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.max_pool3d_with_indices",
        (nn_ops.aten_max_pool3d_with_indices),
        input_wrangler=_max_pool_input_wrangler,
        trace_only=True,
        skips_or_fails=(
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
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.scaled_dot_product_attention",
        nn_ops.aten_scaled_dot_product_attention,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.scaled_dot_product_attention",
                reason="fixme: ORT crashes on Windows, segfaults randomly on Linux",
            ),
            skip(
                "nn.functional.scaled_dot_product_attention",
                matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask"))
                is not None
                and attn_mask.dtype == torch.bool,
                reason="this overload takes a non-boolean mask",
            ),
            skip(
                "nn.functional.scaled_dot_product_attention",
                matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
                reason="dropout is random so the results do not match",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.scaled_dot_product_attention_bool_mask",
        nn_ops.aten_scaled_dot_product_attention_bool_mask,
        trace_only=True,
        skips_or_fails=(
            skip(
                "nn.functional.scaled_dot_product_attention_bool_mask",
                reason="fixme: ORT crashes on Windows, segfaults randomly on Linux",
            ),
            skip(
                "nn.functional.scaled_dot_product_attention_bool_mask",
                matcher=lambda sample: (attn_mask := sample.kwargs.get("attn_mask"))
                is not None
                and attn_mask.dtype != torch.bool,
                reason="this overload takes a boolean mask",
            ),
            skip(
                "nn.functional.scaled_dot_product_attention_bool_mask",
                matcher=lambda sample: sample.kwargs.get("dropout_p") != 0.0,
                reason="dropout is random so the results do not match",
            ),
        ),
    ),
    TorchLibOpInfo(
        "nn.functional.upsample_bilinear2d",
        (nn_ops.aten_upsample_bilinear2d),
        input_wrangler=_upsample_bilinear2d_input_wrangler,
        trace_only=True,
    ),
    TorchLibOpInfo(
        "nn.functional.upsample_nearest2d",
        (nn_ops.aten_upsample_nearest2d),
        input_wrangler=_upsample_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "nn.functional.upsample_nearest2d",
                reason="fixme: ORT fails with invalid model: 'INVALID_ARGUMENT : Failed to load model with error: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)'",
                test_class_name="TestOutputConsistencyFullGraph",
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
        ),
    ),
    TorchLibOpInfo("ones_like", core_ops.aten_ones_like, trace_only=True),
    TorchLibOpInfo(
        "scatter_reduce",
        (core_ops.aten_scatter_reduce),
        input_wrangler=_scatter_reduce_input_wrangler,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "scatter_reduce",
                variant_name="mean",
                reason="ONNX doesn't support reduce='mean' option",
            ),
            skip(
                "scatter_reduce",
                # ONNX has not include_self parameter and default is include_self=True mode
                matcher=lambda sample: sample.kwargs.get("include_self") is False,
                reason="ONNX does't support include_self=False option",
            ),
        ),
    ),
    TorchLibOpInfo("slice_scatter", core_ops.aten_slice_scatter, trace_only=True),
    TorchLibOpInfo("slice", core_ops.aten_slice, trace_only=True),
    TorchLibOpInfo(
        "aten.stft",
        core_ops.aten_stft,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "aten.stft",
                dtypes=[torch.float16],
                reason="RuntimeError: MKL FFT doesn't support tensors of type: Half",
            ),
        ),
    ),  # Custom from extra_opinfo
    TorchLibOpInfo(
        "sum",
        (core_ops.aten_sum_dim_IntList),
        input_wrangler=_sum_input_wrangler,
        trace_only=True,
    ),
    TorchLibOpInfo("transpose", core_ops.aten_transpose, trace_only=True),
    TorchLibOpInfo(
        "var_mean",
        core_ops.aten_var_mean,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "var_mean",
                reason="fixme: Inferred shape and existing shape differ in rank",
            ),
            skip(
                "var_mean",
                variant_name="unbiased",
                reason="fixme: Inferred shape and existing shape differ in rank",
            ),
            xfail(
                "var_mean",
                # kwargs is empty
                matcher=lambda sample: len(sample.kwargs) > 0,
                reason="this Aten overload only support input[0]=tensor and input[1]=bool as input without any kwargs",
            ),
        ),
    ),
    TorchLibOpInfo(
        "var_mean_dim",
        core_ops.aten_var_mean_dim,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "var_mean_dim",
                # kwargs["dim"] must exist, kwargs["correction"] must not exist
                matcher=lambda sample: not (
                    sample.kwargs.get("dim", None) is not None
                    and sample.kwargs.get("correction", None) is None
                ),
                reason="this Aten overload only support with 'dim' argument and without 'correction' argument",
            ),
        ),
    ),
    TorchLibOpInfo(
        "var_mean_correction",
        core_ops.aten_var_mean_correction,
        trace_only=True,
        skips_or_fails=(
            xfail(
                "var_mean_correction",
                reason="fixme: Inferred shape and existing shape differ in rank",
            ),
            skip(
                "var_mean_correction",
                # Don't accept input[1]=bool and 'correction' must be in kwargs
                matcher=lambda sample: len(sample.args) > 0
                or "correction" not in sample.kwargs,
                reason="this Aten overload only support when correction attribute exists",
            ),
        ),
    ),
    TorchLibOpInfo("zeros_like", core_ops.aten_zeros_like, trace_only=True),
)


# Split the scripted and traced ops to make sure we don't forget to script an op
OPINFO_FUNCTION_MAPPING_SCRIPTED: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
] = {
    info.op_info_name: info.op
    for info in TESTED_TORCHLIB_OPS
    if not info.trace_only and not info.complex
}


OPINFO_FUNCTION_MAPPING_TRACE_ONLY: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
] = {
    info.op_info_name: info.op
    for info in TESTED_TORCHLIB_OPS
    if info.trace_only and not info.complex
}

# These ops are not deterministic, so we check shape and dtype only
NONDETERMINISTIC_OPS: frozenset[str] = frozenset(
    info.op_info_name for info in TESTED_TORCHLIB_OPS if info.nondeterministic
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

EXPECTED_SKIPS_OR_FAILS: tuple[ops_test_common.DecorateMeta, ...] = tuple(
    functools.reduce(
        # Flatten the list
        lambda a, b: [*a, *b],
        [
            [meta for meta in info.skips_or_fails if meta.matcher is None]
            for info in TESTED_TORCHLIB_OPS
        ],
    )
)

SKIP_XFAIL_SUBTESTS: tuple[ops_test_common.DecorateMeta, ...] = tuple(
    functools.reduce(
        # Flatten the list
        lambda a, b: [*a, *b],
        [
            [meta for meta in info.skips_or_fails if meta.matcher is not None]
            for info in TESTED_TORCHLIB_OPS
        ],
    )
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

ops_test_common.duplicate_opinfo(OPS_DB, "atleast_1d", ("atleast_1d_single_tensor",))
ops_test_common.duplicate_opinfo(OPS_DB, "atleast_2d", ("atleast_2d_single_tensor",))
ops_test_common.duplicate_opinfo(OPS_DB, "atleast_3d", ("atleast_3d_single_tensor",))


ops_test_common.duplicate_opinfo(OPS_DB, "full_like", ("full_like_dtype",))

ops_test_common.duplicate_opinfo(OPS_DB, "index_put", ("index_put_bool",))

ops_test_common.duplicate_opinfo(OPS_DB, "mean", ("mean_dim",))

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

# MARK: Complex supported functions
COMPLEX_FUNCTION_MAPPING: dict[
    str,
    Callable[..., Any] | tuple[Callable[..., Any], Callable[..., Any]],
] = {info.op_info_name: info.op for info in TESTED_TORCHLIB_OPS if info.complex}


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

# List for different input dtype testing flag
# Added Cast inside below functions so they can support all real dtypes naturally
# -- isfinite, isinf, isneginf, isposinf
# Note:
# Converter fp32 to fp16 model is a significant feature for end users,
# another approach is to add cast before/after the function call,
# that way we also need a list to remember which function need cast
OPINFO_FUNCTION_TARGET_DTYPE: dict[
    str,
    tuple[Any, ...],
] = {
    "add": (
        torch.float32,
        # torch.float16,  # FIXME: float16 failed, tensor-likes are not close for FullGraph mode
        # using https://github.com/microsoft/onnxruntime/issues/15977 to track
    ),
    "chunk": (
        torch.float32,
        # torch.float16,  # FIXME: SplitToSequence op inference failed
        # using https://github.com/microsoft/onnxruntime/issues/16006 to track
    ),
    "col2im": (
        torch.float32,
        # torch.float16,  # FIXME: Tensor-likes are not close
        # using https://github.com/microsoft/onnxruntime/issues/16007 to track
    ),
    "log10": (
        torch.float32,
        # py310-torch-nightly,Shape inference error(s): (op_type:Div, node name: n3): B has inconsistent type tensor(float)
        # torch.float16,
    ),
    "log_softmax": (
        torch.float32,
        # torch.float16,  # FIXME: ORT failed.
    ),
    "log2": (
        torch.float32,
        # windows-latest, py310-torch-nightly, RuntimeError: Unable to create onnxruntime InferenceSession for executing .Div op with onnx model
        # torch.float16,
    ),
    "native_group_norm": (
        torch.float32,
        # torch.float16,  # "GroupNormKernelImpl" not implemented for 'Half' in nightly and weekly
    ),
    "nn.functional.adaptive_avg_pool3d": (
        torch.float32,
        # torch.float16,  # FIXME: ORT inference error GlobalAveragePool
    ),
    "nn.functional.elu": (
        torch.float32,
        # torch.float16,  # ONNX Runtime aborted, ubuntu, py310 torch-nightly
    ),
    "nn.functional.gelu": (
        torch.float32,
        # torch.float16,  # ubuntu py310 torch-nightly failed, ONNX Runtime aborted
    ),
    "nn.functional.relu": (
        torch.float32,
        # ORT cannot support relu in float16
        # file issue: https://github.com/microsoft/onnxruntime/issues/16069
        # torch.float16,
    ),
    "nn.functional.relu6": (
        torch.float32,
        # ORT cannot support relu in float16
        # file issue: https://github.com/microsoft/onnxruntime/issues/16069
        # torch.float16,
    ),
    "nn.functional.selu": (
        torch.float32,
        # torch.float16,  # ubuntu py310 torch-nightly failed, ONNX Runtime aborted
    ),
    "normal": (
        torch.float32,
        # torch.float16,  # FIXME: RandomNormal in ORT failed
    ),
    "randn": (
        torch.float32,
        # torch.float16,  # FIXME: shape inference error
    ),
    "scatter_reduce": (
        torch.float32,
        # torch.float16,  # FIXME: ORT failed
    ),
    "scatter_add": (
        torch.float32,
        # torch.float16,  # FIXME" ORT failed
    ),
    "softmax": (
        torch.float32,
        # torch.float16,  # FIXME: ORT failed
    ),
    "split_with_sizes": (
        torch.float32,
        # torch.float16,  # FIXME: ORT failed
    ),
    "split": (
        torch.float32,
        # torch.float16,  # ORT failed
    ),
    "var_mean": (
        torch.float32,
        # py31--torch-nightly, Unable to create onnxruntime InferenceSession for executing .Mul op with onnx model
        # torch.float16,
    ),
    "var_mean_dim": (
        torch.float32,
        # py310-torch-nightly, FullGraph, AssertionError in ORT
        # torch.float16,
    ),
    "var_mean_correction": (
        torch.float32,
        # py310-onnx-weekly, FullGraph, AssertionError in ORT
        # torch.float16,
    ),
}
