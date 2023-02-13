"""Test op correctness by comparing with PyTorch results."""
from __future__ import annotations

import copy
import dataclasses
import unittest
import warnings
from typing import Any, Callable, Collection, Iterable, Optional, Sequence, TypeVar

import numpy as np
import onnx
import parameterized
import torch
from torch.testing._internal import common_device_type, common_methods_invocations
from torch.testing._internal.opinfo import core as opinfo_core
from torch.utils import _pytree as pytree

import onnxscript
from onnxscript.function_libs.torch_aten.ops import core as core_ops
from onnxscript.function_libs.torch_aten.ops import nn as nn_ops
from onnxscript.function_libs.torch_aten.ops import special as special_ops
from onnxscript.tests.common import version_utils
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


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
) -> DecorateMeta:
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        dtypes=dtypes,
        reason=reason,
    )


def skip(
    op_name: str,
    variant_name: str = "",
    *,
    reason: str,
    dtypes: Optional[Collection[torch.dtype]] = None,
    matcher: Optional[Callable[[Any], Any]] = None,
) -> DecorateMeta:
    """Skips an OpInfo test.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        dtypes: The dtypes to skip.
        reason: The reason for skipping.
        matcher: A function that matches the test sample input. It is used only when
            xfail is in the SKIP_SUBTESTS list.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Skip: {reason}"),
        dtypes=dtypes,
        reason=reason,
        matcher=matcher,
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
            test_class_name,
            base_test_name,
            dtypes=decorate_meta.dtypes,
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
                    warnings.warn(f"OpInfo {new_name} already exists in the database.")
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


def _cat_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Remove the self argument
    if len(args) == 2:
        kwargs["dim"] = args.pop()
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


def _full_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    # Remove the self argument
    if version_utils.torch_older_than("2.0"):
        args.pop(0)
    return args, kwargs


def _upsample_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if "scale_factor" in kwargs:
        kwargs["scales_h"] = kwargs["scale_factor"]
        kwargs["scales_w"] = kwargs["scale_factor"]
        del kwargs["scale_factor"]
    if "size" in kwargs:
        kwargs["size"] = np.array(kwargs["size"])
    return args, kwargs


def _sum_input_wrangler(
    args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if kwargs.get("dim") is not None:
        kwargs["dim"] = np.array(kwargs["dim"], dtype=np.int64)
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
    "abs": core_ops.aten_abs,
    "acos": core_ops.aten_acos,
    "acosh": core_ops.aten_acosh,
    "add": core_ops.aten_add,
    "addmm": core_ops.aten_addmm,
    # "alias": core_ops.aten_alias,  # alias is not in OP-TEST-DB
    "asin": core_ops.aten_asin,
    "asinh": core_ops.aten_asinh,
    "atan": core_ops.aten_atan,
    "atanh": core_ops.aten_atanh,
    "baddbmm": core_ops.aten_baddbmm,
    "bmm": core_ops.aten_bmm,
    "broadcast_to": core_ops.aten_broadcast_to,
    "cat": (core_ops.aten_cat, _cat_input_wrangler),
    "ceil": core_ops.aten_ceil,
    "clamp_max": core_ops.aten_clamp_max,
    "clamp_min": core_ops.aten_clamp_min,
    "clone": core_ops.aten_clone,
    "cos": core_ops.aten_cos,
    "cosh": core_ops.aten_cosh,
    # "detach": core_ops.aten_detach,  # detach is not in OP-TEST-DB
    "div": core_ops.aten_div,
    "dot": core_ops.aten_dot,
    "empty": core_ops.aten_empty,
    "eq": core_ops.aten_eq,
    "equal": core_ops.aten_equal,
    "exp": core_ops.aten_exp,
    "exp2": core_ops.aten_exp2,
    "expand": core_ops.aten_expand,
    "erf": core_ops.aten_erf,
    "fmod": core_ops.aten_fmod,
    "full": (core_ops.aten_full, _full_input_wrangler),
    "full_like": core_ops.aten_full_like,
    "ge": core_ops.aten_ge,
    "gt": core_ops.aten_gt,
    "isinf": core_ops.aten_isinf,
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
    "minimum": core_ops.aten_minimum,
    "mm": core_ops.aten_mm,
    "mul": core_ops.aten_mul,
    "ne": core_ops.aten_ne,
    "neg": core_ops.aten_neg,
    "new_full": core_ops.aten_new_full,
    "nn.functional.adaptive_avg_pool1d": nn_ops.aten_adaptive_avg_pool1d,
    "nn.functional.adaptive_avg_pool2d": nn_ops.aten_adaptive_avg_pool2d,
    "nn.functional.adaptive_avg_pool3d": nn_ops.aten_adaptive_avg_pool3d,
    "nn.functional.celu": nn_ops.aten_celu,
    "nn.functional.elu": nn_ops.aten_elu,
    "nn.functional.embedding": (core_ops.aten_embedding, _embedding_input_wrangler),
    "nn.functional.leaky_relu": nn_ops.aten_leaky_relu,
    "nn.functional.logsigmoid": nn_ops.aten_log_sigmoid,
    "nn.functional.relu": nn_ops.aten_relu,
    "nn.functional.relu6": nn_ops.aten_relu6,
    "nn.functional.selu": core_ops.aten_selu,
    "nn.functional.upsample_nearest2d": (
        nn_ops.aten_upsample_nearest2d,
        _upsample_input_wrangler,
    ),
    "nonzero": core_ops.aten_nonzero,
    "ones": core_ops.aten_ones,
    "permute": core_ops.aten_permute,
    "pow": core_ops.aten_pow,
    "reciprocal": core_ops.aten_reciprocal,
    "remainder": core_ops.aten_remainder,
    "repeat": core_ops.aten_repeat,
    "reshape": core_ops.aten_reshape,
    "round": core_ops.aten_round,
    "rsqrt": core_ops.aten_rsqrt,
    "rsub": core_ops.aten_rsub,
    "select": core_ops.aten_select,
    "sigmoid": core_ops.aten_sigmoid,
    "sign": core_ops.aten_sign,
    "sin": core_ops.aten_sin,
    "sinh": core_ops.aten_sinh,
    "softmax": special_ops.aten_special_softmax,
    "split": core_ops.aten_split,
    "sqrt": core_ops.aten_sqrt,
    "sub": core_ops.aten_sub,
    "t": core_ops.aten_t,
    "tan": core_ops.aten_tan,
    "tanh": core_ops.aten_tanh,
    "topk": core_ops.aten_topk,
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
    "amax": core_ops.aten_amax,
    "amin": core_ops.aten_amin,
    "arange_start_step": core_ops.aten_arange_start_step,
    "arange_start": core_ops.aten_arange_start,
    "arange": core_ops.aten_arange,
    "argmax": core_ops.aten_argmax,
    "argmin": core_ops.aten_argmin,
    "clamp": core_ops.aten_clamp,
    "cumsum": core_ops.aten_cumsum,
    "convolution": core_ops.aten_convolution,
    "empty_like": core_ops.aten_empty_like,
    "index_select": core_ops.aten_index_select,
    "native_layer_norm": core_ops.aten_native_layer_norm,
    "nn.functional.conv1d": core_ops.aten_conv1d,
    "nn.functional.conv2d": core_ops.aten_conv2d,
    "nn.functional.conv3d": core_ops.aten_conv3d,
    "nn.functional.gelu": nn_ops.aten_gelu,
    "nn.functional.linear": nn_ops.aten_linear,
    "ones_like": core_ops.aten_ones_like,
    "slice": core_ops.aten_slice,
    "sum": (core_ops.aten_sum_dim_IntList, _sum_input_wrangler),
    "transpose": core_ops.aten_transpose,
    "zeros_like": core_ops.aten_zeros_like,
}

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
    skip("empty", reason="Using zeros to simulate empty"),
    skip("empty_like", reason="Using zeros_like to simulate empty_like"),
    xfail("logcumsumexp", reason="naive implementation not numerically stable"),
    xfail("logsumexp", reason="ONNX Runtime 1.13 does not support ReduceLogSumExp-18"),
    xfail(
        "nn.functional.upsample_nearest2d",
        reason="enable when ONNX Runtime does support opset18",
    ),
    xfail("round", variant_name="decimals_0", reason="The op does not support decimals"),
    xfail("round", variant_name="decimals_3", reason="The op does not support decimals"),
    xfail("round", variant_name="decimals_neg_3", reason="The op does not support decimals"),
)


SKIP_SUBTESTS: tuple[DecorateMeta, ...] = (
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
        "expand",
        matcher=lambda sample: (np.array(sample.args[0]) > 0).all() is np.bool_(False),
        reason="Negative value is not supported",
    ),
    skip(
        "nonzero",
        matcher=lambda sample: sample.kwargs.get("as_tuple") is not None,
        reason="as_tuple=True is not supported",
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
)

duplicate_opinfo(
    OPS_DB,
    "arange",
    (
        "arange_start",
        "arange_start_step",
    ),
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

duplicate_opinfo(OPS_DB, "new_full", ("full",))

# END OF SECTION TO MODIFY #####################################################


OP_WITH_SKIPPED_SUBTESTS = frozenset(meta.op_name for meta in SKIP_SUBTESTS)
ALL_OPS_IN_DB = frozenset(op_info.name for op_info in OPS_DB)
# Assert all ops in OPINFO_FUNCTION_MAPPING are in the OPS_DB
assert TESTED_OPS.issubset(ALL_OPS_IN_DB), f"{TESTED_OPS - ALL_OPS_IN_DB} not in OPS_DB"


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


class TestFunctionValidity(unittest.TestCase):
    def test_all_script_functions_are_onnx_functions(self):
        for func_with_wrangler in OPINFO_FUNCTION_MAPPING_SCRIPTED.values():
            if isinstance(func_with_wrangler, tuple):
                func = func_with_wrangler[0]
            else:
                func = func_with_wrangler
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


class TestOutputConsistency(unittest.TestCase):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    @common_device_type.ops(  # type: ignore[misc]
        [info for info in OPS_DB if info.name in TESTED_OPS],
        allowed_dtypes=TESTED_DTYPES,
    )
    @add_decorate_info(
        OPS_DB,
        "TestOutputConsistency",
        "test_output_match",
        skip_or_xfails=EXPECTED_SKIPS_OR_FAILS,
    )
    def test_output_match(self, device: str, dtype: torch.dtype, op):
        """Base test method for testing each opset, used by instantiate_device_type_tests."""
        # device is provided by instantiate_device_type_tests, but we only want to run in cpu.
        assert device == "cpu"

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

        for (i, cpu_sample) in enumerate(samples):
            inputs = (cpu_sample.input, *cpu_sample.args)
            # Provide the repr to subtest because tensors are not serializable in parallel test runs
            with self.subTest(
                sample_num=i,
                inputs=repr(inputs),
                kwargs=repr(cpu_sample.kwargs),
            ):
                skip_reason = _should_skip_test_sample(op.name, cpu_sample)
                if skip_reason is not None:
                    # Cannot use self.skip because pytest would skip the entire test
                    warnings.warn(f"skipped sample {i}. Reason: {skip_reason}")
                    continue
                input_onnx = [_convert_tensor_to_numpy(x) for x in inputs]
                kwargs_onnx = _convert_kwargs_for_onnx(cpu_sample.kwargs)
                if input_wrangler:
                    input_onnx, kwargs_onnx = input_wrangler(input_onnx, kwargs_onnx)
                torch_output = op(*inputs, **cpu_sample.kwargs)
                function_output = onnx_function(*input_onnx, **kwargs_onnx)

                # TODO: add pytree structure comparison.
                flattened_torch_outputs, _ = pytree.tree_flatten(torch_output)
                flattened_function_outputs, _ = pytree.tree_flatten(function_output)

                assert flattened_torch_outputs
                assert len(flattened_torch_outputs) == len(flattened_function_outputs)

                for torch_output, function_output in zip(
                    flattened_torch_outputs, flattened_function_outputs
                ):
                    if dtype == torch.float32:
                        # Relax atol and rtol for float32 based on empirical results
                        # The current most relaxed values are for aten::native_layer_norm
                        rtol = 3.7e-5
                        atol = 1.8e-4
                    else:
                        rtol = None
                        atol = None

                    if not isinstance(function_output, np.ndarray):
                        # An onnxscript tensor
                        function_output = function_output.value

                    # Use torch.testing as opposed to np.testing to ensure dtypes and shapes match
                    torch.testing.assert_close(
                        torch.tensor(function_output).cpu(),
                        torch_output.cpu()
                        if isinstance(torch_output, torch.Tensor)
                        else torch.tensor(torch_output).cpu(),
                        rtol=rtol,
                        atol=atol,
                    )


common_device_type.instantiate_device_type_tests(
    TestOutputConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    unittest.main()
