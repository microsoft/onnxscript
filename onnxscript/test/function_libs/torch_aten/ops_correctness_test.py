"""Test op correctness by comparing with PyTorch results."""
from __future__ import annotations

import copy
import dataclasses
import unittest
from typing import Any, Callable, Collection, Iterable, Optional, Sequence, TypeVar

import numpy as np
import onnxruntime.capi.onnxruntime_pybind11_state
import torch
from torch.testing._internal import common_device_type, common_methods_invocations
from torch.testing._internal.opinfo import core as opinfo_core

import onnxscript
from onnxscript.function_libs.torch_aten.ops import core as core_ops
from onnxscript.function_libs.torch_aten.ops import nn as nn_ops

T = TypeVar("T")

SUPPORTED_DTYPES = (
    # Boolean
    torch.bool,
    # Integers
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # Floating types
    torch.float16,
    torch.float32,
    torch.float64,
)

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
    return tuple(dtype for dtype in SUPPORTED_DTYPES if dtype not in dtypes)


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


def xfail(
    op_name: str,
    variant_name: str = "",
    *,
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: Optional[str] = None,
):
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        dtypes: The dtypes to expect the failure.
        reason: The reason for the failure.
    """
    if reason is None:
        raise ValueError("Please specify a reason.")
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
    dtypes: Optional[Collection[torch.dtype]] = None,
    reason: Optional[str] = None,
):
    """Skips an OpInfo test.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        dtypes: The dtypes to skip.
        reason: The reason for skipping.
    """
    if reason is None:
        raise ValueError("Please specify a reason.")
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip(f"Don't care: {reason}"),
        dtypes=dtypes,
        reason=reason,
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


# Modify this section ##########################################################

# Ops to be tested for numerical consistency between onnx and pytorch
OPINFO_FUNCTION_MAPPING = {
    "add": core_ops.aten_add,
    "mul": core_ops.aten_mul,
    "nn.functional.elu": nn_ops.aten_elu,
    "nn.functional.relu6": nn_ops.aten_relu6,
    "nn.functional.selu": core_ops.aten_selu,
    "sub": core_ops.aten_sub,
}

TESTED_OPS = frozenset(OPINFO_FUNCTION_MAPPING)

EXPECTED_SKIPS_OR_FAILS = (
    xfail("add", dtypes=BOOL_TYPES, reason="Add is not defined on bool tensors"),
    xfail("mul", dtypes=BOOL_TYPES, reason="Mul is not defined on bool tensors"),
    xfail(
        "nn.functional.elu",
        dtypes=dtypes_except(torch.float16, torch.float32),
        reason="ONNX Runtime doesn't support float64 for Elu",
    ),
    xfail(
        "nn.functional.relu6",
        dtypes=dtypes_except(torch.float16, torch.float32),
        reason="ONNX Runtime doesn't support float64 for Relu",
    ),
    xfail(
        "nn.functional.selu",
        dtypes=dtypes_except(torch.float16, torch.float32),
        reason="ONNX Runtime doesn't support float64 for Selu",
    ),
    xfail("sub", dtypes=BOOL_TYPES, reason="Sub is not defined on bool tensors"),
)
# END OF SECTION TO MODIFY #####################################################


OPS_DB = copy.deepcopy(common_methods_invocations.op_db)


class TestOutputConsistency(unittest.TestCase):
    """Test output consistency between exported ONNX models and PyTorch eager mode.

    This is a parameterized test suite.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    @common_device_type.ops(  # type: ignore[misc]
        [info for info in OPS_DB if info.name in TESTED_OPS],
        allowed_dtypes=SUPPORTED_DTYPES,
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

        onnx_function = OPINFO_FUNCTION_MAPPING[op.name]
        scripted_function = onnxscript.script()(onnx_function)

        for (i, cpu_sample) in enumerate(samples):
            inputs = (cpu_sample.input, *cpu_sample.args)
            # Provide the repr to subtest because tensors are not serializable in parallel test runs
            with self.subTest(
                sample_num=i,
                inputs=repr(inputs),
                kwargs=repr(cpu_sample.kwargs),
            ):
                input_numpy = [x.numpy() for x in inputs if isinstance(x, torch.Tensor)]
                torch_output = op(*inputs, **cpu_sample.kwargs)
                try:
                    function_output = scripted_function(*input_numpy, **cpu_sample.kwargs)
                # pylint: disable=c-extension-no-member
                except onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented:
                    self.skipTest(
                        f"ONNX Runtime doesn't support running {op.name} with dtype {dtype}",
                    )
                # pylint: enable=c-extension-no-member

                # Use torch testing to ensure dtypes and shapes match
                torch.testing.assert_close(
                    torch.tensor(function_output),
                    torch_output,
                )


common_device_type.instantiate_device_type_tests(
    TestOutputConsistency, globals(), only_for="cpu"
)


if __name__ == "__main__":
    unittest.main()
