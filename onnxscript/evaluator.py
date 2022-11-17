# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import abc
import contextlib
import pprint
from typing import Any, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
)

from onnxscript import autocast, irbuilder, onnx_opset, tensor, utils, values


class Evaluator(abc.ABC):
    """Base class for evaluation of ONNX ops.

    The execution of onnxscript functions in eager-mode is dispatched to an Evaluator
    instance (or, more precisely, to the eval method of the Evaluator instance).
    The evaluator is expected to transform the input/output/attribute representation
    supported by onnxscript to those expected by a particular backend.
    """

    def eval(self, schema, inputs, attributes):
        closure = self.adapt_attributes(schema, attributes)
        inputs = self.adapt_inputs(schema, inputs)
        outputs = self._eval(schema, inputs, attributes, closure)
        return self.adapt_outputs(schema, outputs)

    def adapt_inputs(self, schema, inputs):
        """Transform inputs to the expected format for the evaluator.

        Enables some syntactic sugar, such as the use of Python scalars,
        in a manner consistent with the translator. See autocast.py for details.
        """
        return autocast.dynamic_cast_inputs(schema, *inputs)

    def adapt_attributes(self, schema, attributes):
        """Transform attributes (in-place) to the expected format for the evaluator.

        Returns a closure that can be used to evaluate graph-valued attributes.
        """
        use_graph_attribute = self.use_graph_attribute(schema)
        closure = {}
        for k, v in attributes.items():
            if isinstance(v, values.OnnxClosure):
                if use_graph_attribute:
                    attributes[k] = v.function_ir.to_graph_proto()
                    for pyvar, onnxvar in v.function_ir.outer_scope_variables:
                        closure[onnxvar.value] = v.frame.f_locals[pyvar]
                else:
                    attributes[k] = v.function
            elif callable(v):
                raise ValueError(
                    f"Error: function-valued attribute {v.__name__} has no graph_proto"
                    "attribute. Did you forget to decorate it with @graph?"
                )
        return closure

    def adapt_outputs(self, schema, outputs):  # pylint: disable=unused-argument
        """Adapt evaluator's output to convention used in onnxscript.

        Onnxscript uses a tuple/sequence only when number of outputs > 1.
        """
        return outputs[0] if len(outputs) == 1 else outputs

    def use_graph_attribute(self, schema):  # pylint: disable=unused-argument
        return True

    @abc.abstractmethod
    def _eval(self, schema, inputs, attributes, closure):
        pass


# Utilities for evaluation using ORT:


class EagerModeError(RuntimeError):
    pass


def _rename_io(prefix, i, arg):
    if arg is None:
        return ""
    return f"{prefix}{i}"


def compute_num_outputs(schema, *args, **kwargs):
    """Returns the number of outputs expected.
    TODO: Use ONNX type inference to replace the special-case handling below.
    """
    if schema.domain == "":
        if schema.name == "BatchNormalization":
            if not kwargs.get("training_mode", 0):
                return 1
        if schema.name == "LSTM":
            return 3
        if schema.name == "Split":
            if len(args) == 1:
                raise EagerModeError(
                    "Operator Split: the number of expected outputs defines the split. "
                    "This information is unknown here."
                )
        if schema.name == "Scan":
            scan_body = kwargs["body"]
            return len(scan_body.output)
        if schema.name == "Loop":
            loop_body = kwargs["body"]
            return len(loop_body.output) - 1
    return len(schema.outputs)


_cache_models: dict[Any, ort.InferenceSession] = {}


def _cache_(model, providers):
    serialized = model.SerializeToString()
    key = serialized, tuple(providers)
    if key in _cache_models:
        return _cache_models[key]
    session = ort.InferenceSession(serialized, providers=providers)
    _cache_models[key] = session
    return session


def os_to_ort_value(v):
    """Converts an onnxscript encoding of an ONNX value into the encoding used by ORT."""
    if isinstance(v, tensor.Tensor):
        return v.value
    if isinstance(v, list):
        return v
    if v is None:
        # Treated as a static-optional value.
        # Dynamic optional None not yet supported.
        return v
    if isinstance(v, np.ndarray):
        return v
    raise TypeError(f"Unexpected ORT value type {type(v)}.")


def ort_to_os_value(v):
    """Converts an ORT encoding of an ONNX value into the encoding used by onnxscript."""
    if isinstance(v, np.ndarray):
        return tensor.Tensor(v)
    if isinstance(v, list):
        return v
    if v is None:
        raise TypeError("Dynamic optional values not yet supported.")
    raise TypeError(f"Unexpected ORT value type {type(v)}.")


def call_ort(schema, args, kwargs, implicit_args=None):
    implicit_args = implicit_args or {}
    # Convert input values to ORT representation-type:
    args = [os_to_ort_value(x) for x in args]
    implicit_args = {k: os_to_ort_value(v) for k, v in implicit_args.items()}

    # Construct ONNX model with a single op call:
    inputs = [_rename_io("input", i, arg) for i, arg in enumerate(args)]

    num_outputs = compute_num_outputs(schema, *args, **kwargs)
    outputs = [f"output{str(i)}" for i in range(num_outputs)]

    node = onnx.helper.make_node(schema.name, inputs, outputs, domain=schema.domain, **kwargs)
    input_value_infos = utils.values_to_value_infos(zip(inputs, args))
    implicit_value_infos = utils.values_to_value_infos(implicit_args.items())
    output_value_infos = [
        onnx.helper.make_value_info(name, onnx.TypeProto()) for name in outputs
    ]

    graph = onnx.helper.make_graph(
        [node], "node_graph", input_value_infos + implicit_value_infos, output_value_infos
    )
    opset_id = onnx.helper.make_opsetid(schema.domain, schema.since_version)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[opset_id],
        ir_version=irbuilder.select_ir_version(schema.since_version, domain=schema.domain),
    )
    model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)
    try:
        session = _cache_(model, ["CPUExecutionProvider"])
    except (Fail, InvalidGraph, InvalidArgument) as e:
        raise RuntimeError(
            f"Unable to create onnxruntime InferenceSession "
            f"for executing {schema.domain}.{schema.name} op "
            f"with onnx model\n{utils.proto2text(model)}"
        ) from e

    session_run_input = {name: arg for name, arg in zip(inputs, args) if name != ""}
    session_run_input.update(implicit_args)

    try:
        result = session.run(None, session_run_input)
    except (RuntimeError, Fail) as e:
        raise RuntimeError(
            f"Unable to execute model operator {schema.name!r} due to {e!r}"
            f"\ninput types:\n"
            f"{pprint.pformat({k: type(v) for k, v in zip(inputs, args)})}"
            f"\nmodified input types:\n"
            f"{pprint.pformat({k: type(v) for k, v in session_run_input.items()})}"
            f"\ninputs:\n{pprint.pformat(session_run_input)}\n{model}"
        ) from e

    # Map ORT output values to the onnxscript representation-type.
    return [ort_to_os_value(x) for x in result]


def schema_id(schema):
    return schema.name, schema.domain, schema.since_version


class ORTEvaluator(Evaluator):
    """Evaluates ONNX ops using ONNX Runtime."""

    def _eval(self, schema, inputs, attributes, closure):
        return call_ort(schema, inputs, attributes, closure)


ort_evaluator = ORTEvaluator()


class ORTMixedEvaluator(ORTEvaluator):
    """Evaluates ONNX ops using ONNX Runtime, unless an overriding python implementation
    is registered. This is useful for higher-order ops such as Scan and SequenceMap,
    allowing for python-based debugging.
    """

    def __init__(self) -> None:
        super().__init__()
        self._python_ops: dict[Any, Any] = {}

    def use_graph_attribute(self, schema):
        return schema_id(schema) not in self._python_ops

    def _eval(self, schema, inputs, attributes, closure):
        schemaid = schema_id(schema)
        if schemaid in self._python_ops:
            return self._python_ops[schemaid](inputs, attributes)
        else:
            return super()._eval(schema, inputs, attributes, closure)

    def register(self, opset: Optional[values.Opset] = None):
        opset = opset or onnx_opset.default_opset

        def decorator(function):
            schema = opset[function.__name__]
            self._python_ops[schema_id(schema)] = function
            return function

        return decorator


ort_mixed_evaluator = ORTMixedEvaluator()


@ort_mixed_evaluator.register()
def SequenceMap(inputs, attributes):
    """Evaluates a SequenceMap op."""
    fun = attributes["body"]

    def get_input_of(input_index, iter_num):
        input = inputs[input_index]
        if isinstance(input, list):
            return input[iter_num]
        return input

    def get_input(iter_num):
        return [get_input_of(input_index, iter_num) for input_index in range(len(inputs))]

    return [fun(*(get_input(i))) for i in range(len(inputs[0]))]


# Used to control the default evaluator instance. A simple approach for now.

_default_evaluator: Evaluator = ort_evaluator


def default() -> Evaluator:
    """Returns the default Evaluator default."""
    return _default_evaluator


def set_default(new_default: Evaluator) -> None:
    """Sets the current Evaluator default."""
    global _default_evaluator  # pylint: disable=global-statement
    _default_evaluator = new_default


@contextlib.contextmanager
def default_as(temp_default: Evaluator):
    """Context manager that temporarily switches the default evaluator."""
    old_default = _default_evaluator
    set_default(temp_default)
    try:
        yield
    finally:
        set_default(old_default)


def eval(schema, inputs, attributes):
    """Evaluate using current default evaluator"""
    return default().eval(schema, inputs, attributes)
