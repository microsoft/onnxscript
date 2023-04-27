"""Common utils for testing operators."""

from __future__ import annotations

import contextlib
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
import pytest
import torch
from torch.testing._internal.opinfo import core as opinfo_core

import onnxscript
import onnxscript.evaluator
from onnxscript.function_libs.torch_lib import graph_building

T = TypeVar("T")


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
    test_behavior: str
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
    matcher: Optional[Callable[[Any], Any]] = None,
    enabled_if: bool = True,
    test_class_name: Optional[str] = None,
) -> DecorateMeta:
    """Expects an OpInfo test to fail.

    Args:
        op_name: The name of the operator.
        variant_name: Optional OpInfo variant_test_name.
        reason: The reason for the failure.
        dtypes: The dtypes to expect the failure.
        matcher: A function that matches the test sample input. It is used only when
            the xfail is in the SKIP_XFAIL_SUBTESTS list.
        enabled_if: Whether the xfail is enabled.
        test_class_name: The test class name to apply the xfail to. If None, the
            xfail is applied to all test classes.
    """
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        dtypes=dtypes,
        matcher=matcher,
        reason=reason,
        enabled_if=enabled_if,
        test_class_name=test_class_name,
        test_behavior="xfail",
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
            the skip is in the SKIP_XFAIL_SUBTESTS list.
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
        test_behavior="skip",
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


def convert_tensor_to_numpy(input: Any) -> Any:
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if isinstance(input[0], torch.Tensor):
            return [convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)

    return input


def convert_kwargs_for_onnx(kwargs: dict[str, Any]) -> dict[str, Any]:
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


def graph_executor(
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


def eager_executor(
    outputs,
) -> Callable[[Callable[..., Any], tuple[Any], dict[str, Any]], None]:
    """Eagerly executes a function."""

    del outputs  # Unused

    def executor(function, args, kwargs):
        return function(*args, **kwargs)

    return executor


@contextlib.contextmanager
def normal_xfail_skip_test_behaviors(
    test_behavior: Optional[str] = None, reason: Optional[str] = None
):
    """This context manager is used to handle the different behaviors of xfail and skip.

    Args:
        test_behavior (optional[str]): From DecorateMeta name, can be 'skip', 'xfail', or None.
        reason (optional[str]): The reason for the failure or skip.

    Raises:
        e: Any exception raised by the test case if it's not an expected failure.
    """

    # We need to skip as soon as possible, as SegFault might also be a case.
    if test_behavior == "skip":
        pytest.skip(reason=reason)

    try:
        yield
    # We could use `except (AssertionError, RuntimeError, ...) as e:`, but it needs
    # to go over all test cases to find the right exception type.
    except Exception as e:  # pylint: disable=broad-exception-caught
        if test_behavior is None:
            raise e
        if test_behavior == "xfail":
            pytest.xfail(reason=reason)
    else:
        if test_behavior == "xfail":
            pytest.fail("Test unexpectedly passed")
