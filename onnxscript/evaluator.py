# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import abc
import contextlib
import pprint
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import onnx
import onnx.defs
import onnx.helper
import onnx.reference
from typing_extensions import TypeAlias

from onnxscript import irbuilder, onnx_opset, tensor, values
from onnxscript._internal import autocast, param_manipulation, utils

UserModeValue: TypeAlias = Union[Optional[np.ndarray], Sequence["UserModeValue"]]

EagerModeValue: TypeAlias = Union[Optional["tensor.Tensor"], Sequence["EagerModeValue"]]

ExtendedModeValue: TypeAlias = Union[
    Optional["tensor.Tensor"],
    Sequence["ExtendedModeValue"],
    np.ndarray,
    int,
    float,
    bool,
    str,
]

_T = TypeVar("_T")


def _adapt_to_eager_mode(inputs: ExtendedModeValue) -> tuple[EagerModeValue, bool]:
    """Adapts inputs into representation used by onnxscript eager mode.

    This does the following transformations:
    * It adds an onnxscript Tensor wrapper around numpy arrays, which
    allows the use of overloaded operators like + to be controlled by onnxscript.
    * It also provides a promotion of scalars into tensors as a convenience.
    This is needed to complement the similar promotion supported by the
    onnxscript converter (for example, when an attribute is promoted and used
    as an input argument).

    Args:
        inputs: a list/tuple of inputs to an ONNX function

    Returns:
        a pair (wrapped_inputs, flag) where flag indicates whether any numpy array
        was wrapped into a Tensor.
    """
    has_array = False

    def adapt(input: ExtendedModeValue) -> EagerModeValue:
        if isinstance(input, np.ndarray):
            nonlocal has_array
            has_array = True
            return tensor.Tensor(input)
        if isinstance(input, tensor.Tensor):
            return input
        if isinstance(input, (bool, float)):
            return tensor.Tensor(np.array(input))
        if isinstance(input, int):
            return tensor.Tensor(np.array(input, dtype=np.int64))
        if input is None:
            return None
        if isinstance(input, list):
            return [adapt(elt) for elt in input]
        if isinstance(input, tuple):
            return tuple(adapt(elt) for elt in input)
        raise TypeError(f"Unexpected input type {type(input)}.")

    result = adapt(inputs)
    return result, has_array


def _adapt_to_user_mode(output: ExtendedModeValue) -> UserModeValue:
    """Unwraps Tensor wrapper around numpy arrays.

    Args:
        output: output of an ONNX function, which can be either a single
            onnx value or a list/tuple of onnx values.

    Returns:
        unwrapped output
    """
    if isinstance(output, tensor.Tensor):
        return output.value
    if output is None:
        return None
    if isinstance(output, list):
        return [_adapt_to_user_mode(elt) for elt in output]
    if isinstance(output, tuple):
        return tuple(_adapt_to_user_mode(elt) for elt in output)
    if isinstance(output, np.ndarray):
        return output
    raise TypeError(f"Unexpected type {type(output)}.")


def _unwrap_tensors_in_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Unwrap tensors in a mapping to numpy arrays."""
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k] = v
        if isinstance(v, tensor.Tensor):
            new_kwargs[k] = v.value

    return new_kwargs


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluating ONNX ops."""

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        inputs: Sequence[ExtendedModeValue],
        attributes: Mapping[str, Any],
    ):
        """Evaluates an ONNX op.

        Args:
            schema: The OpSchema of the operator to evaluate.
            inputs: The ONNX inputs to the op.
            attributes: The ONNX attributes to the op.
        """

    def eval_function(
        self,
        function: values.OnnxFunction,
        args: Sequence[ExtendedModeValue],
        kwargs: Mapping[str, ExtendedModeValue],
    ):
        """Evaluates an OnnxFunction.

        Args:
            function: The OnnxFunction to evaluate.
            args: The positional arguments to the function.
            kwargs: The keyword arguments to the function.
        """


class BaseEvaluator(Evaluator, abc.ABC):
    """Base class for evaluation of ONNX ops.

    The execution of onnxscript functions in eager-mode is dispatched to an Evaluator
    instance (or, more precisely, to the eval method of the Evaluator instance).
    The evaluator is expected to transform the input/output/attribute representation
    supported by onnxscript to those expected by a particular backend.
    """

    def __init__(self, ignore_unknown_function_kwargs: bool = False):
        """Initializes a BaseEvaluator.

        Args:
            ignore_unknown_function_kwargs: Whether to ignore unknown keyword arguments
                when evaluating an OnnxFunction. This is useful when using the
                evaluator to validate operators programmatically, where
                additional keyword arguments that is not part of the signature
                may be provided to the function.
        """
        self._ignore_unknown_function_kwargs = ignore_unknown_function_kwargs

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        inputs: Sequence[ExtendedModeValue],
        attributes: Mapping[str, Any],
    ):
        """Evaluates an ONNX op.

        Args:
            schema: The OpSchema of the operator to evaluate.
            inputs: The ONNX inputs to the op.
            attributes: The ONNX attributes to the op.
        """
        attributes = _unwrap_tensors_in_kwargs(attributes)
        attributes, closure = self.adapt_attributes(schema, attributes)
        inputs = self.adapt_inputs(schema, inputs)
        outputs = self._eval(schema, inputs, attributes, closure)
        return self.adapt_outputs(schema, outputs)

    def adapt_inputs(self, schema: onnx.defs.OpSchema, inputs: Sequence[ExtendedModeValue]):
        """Transform inputs to the expected format for the evaluator.

        Enables some syntactic sugar, such as the use of Python scalars,
        in a manner consistent with the translator. See autocast.py for details.
        """
        return autocast.dynamic_cast_inputs(schema, inputs)

    def adapt_attributes(
        self, schema: onnx.defs.OpSchema, attributes: Mapping[str, ExtendedModeValue]
    ) -> tuple[dict[str, ExtendedModeValue], dict[str, ExtendedModeValue]]:
        """Transform attributes to the expected format for the evaluator.

        Returns:
            A closure that can be used to evaluate graph-valued attributes.
        """
        use_graph_attribute = self.use_graph_attribute(schema)
        closure: dict[Any, Any] = {}
        adapted_attributes = {}
        for k, v in attributes.items():
            if isinstance(v, values.OnnxClosure):
                if use_graph_attribute:
                    adapted_attributes[k] = v.function_ir.to_graph_proto()
                    for pyvar, onnxvar in v.function_ir.outer_scope_variables:
                        closure[onnxvar.value] = v.frame.f_locals[pyvar]
                else:
                    adapted_attributes[k] = v.function
            elif callable(v):
                raise TypeError(
                    f"Error: function-valued attribute {v.__name__} has no graph_proto"
                    "attribute. Did you forget to decorate it with @graph?"
                )
            else:
                adapted_attributes[k] = v
        return adapted_attributes, closure

    def adapt_outputs(self, schema: onnx.defs.OpSchema, outputs: Sequence[EagerModeValue]):
        """Adapt evaluator's output to convention used in onnxscript.

        Onnxscript uses a tuple/sequence only when number of outputs > 1.
        """
        del schema  # unused
        return outputs[0] if len(outputs) == 1 else outputs

    def use_graph_attribute(self, schema: onnx.defs.OpSchema):
        del schema  # unused
        return True

    @abc.abstractmethod
    def _eval(
        self,
        schema: onnx.defs.OpSchema,
        inputs: Sequence[ExtendedModeValue],
        attributes: Mapping[str, ExtendedModeValue],
        closure: Mapping[str, ExtendedModeValue],
    ) -> EagerModeValue:
        """Evaluates an ONNX op given its schema and inputs/attributes.

        Args:
            schema: The schema of the op to evaluate.
            inputs: The ONNX inputs to the op.
            attributes: The ONNX attributes to the op.
            closure: The closure to use when evaluating graph-valued attributes.
        """

    def eval_function(
        self,
        function: values.OnnxFunction,
        args: Sequence[ExtendedModeValue],
        kwargs: Mapping[str, ExtendedModeValue],
    ):
        """Evaluates a function in eager mode.

        Override this function to change the evaluator's behavior for functions.

        Args:
            function: The OnnxFunction to evaluate.
            args: The positional arguments to the function.
            kwargs: The keyword arguments to the function.
        """
        param_schemas = function.param_schemas()
        # Split happens in the evaluator instead of the OnnxFunction __call__ method
        # so that evaluators can control behaviors like whether to fill in default values for attributes.
        tagged_args, tagged_kwargs = param_manipulation.tag_arguments_with_param_schemas(
            param_schemas,
            args,
            kwargs,
            fill_defaults=False,
            allow_extra_kwargs=self._ignore_unknown_function_kwargs,
        )

        adapted_args: list[ExtendedModeValue] = []
        adapted_kwargs: dict[str, ExtendedModeValue] = {}
        has_array = False
        for arg, param_schema in tagged_args:
            if param_schema.is_input:
                adapted_arg, _has_array = _adapt_to_eager_mode(arg)
                has_array = has_array or _has_array
                adapted_args.append(adapted_arg)
            else:
                adapted_args.append(arg)

        for key, (arg, param_schema) in tagged_kwargs.items():
            if param_schema.is_input:
                adapted_arg, _has_array = _adapt_to_eager_mode(arg)
                has_array = has_array or _has_array
                adapted_kwargs[key] = adapted_arg
            else:
                adapted_kwargs[key] = arg

        result = function.function(*adapted_args, **adapted_kwargs)

        # We use a heuristic to decide whether to return output values as
        # numpy arrays or tensor.Tensors. If the function has at least one
        # numpy array as input, we return numpy arrays. Otherwise, we return
        # tensor.Tensors. We could use a user-specified flag to control this
        # or explicitly track whether this is a top-level function-call or
        # a nested function-call.

        return _adapt_to_user_mode(result) if has_array else result


# Utilities for evaluation using ORT:


class EagerModeError(RuntimeError):
    pass


def _rename_io(prefix, i, arg):
    if arg is None:
        return ""
    return f"{prefix}{i}"


def compute_num_outputs(
    schema: onnx.defs.OpSchema, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> int:
    """Returns the number of outputs expected."""

    # TODO: Use ONNX type inference to replace the special-case handling below.
    if schema.domain == "":
        if schema.name == "BatchNormalization":
            if not kwargs.get("training_mode", 0):
                return 1
        if schema.name == "LSTM":
            return 3
        if schema.name == "Split":
            if len(args) == 1 and "num_outputs" not in kwargs:
                raise EagerModeError(
                    "Operator Split: the number of expected outputs defines the split. "
                    "This information is unknown here."
                )
            if len(args) == 2:  # has argument(split)
                return len(args[1])
            else:  # no argument(split), check attribute(num_outputs)
                return kwargs["num_outputs"]
        if schema.name == "Scan":
            scan_body = kwargs["body"]
            return len(scan_body.output)
        if schema.name == "Loop":
            loop_body = kwargs["body"]
            return len(loop_body.output) - 1
    return len(schema.outputs)


def _onnxscript_to_numpy_value(v):
    """Converts an onnxscript encoding of an ONNX value into the numpy encoding used by runtimes."""
    if isinstance(v, tensor.Tensor):
        return v.value
    if isinstance(v, list):
        return [_onnxscript_to_numpy_value(x) for x in v]
    if isinstance(v, tuple):
        if len(v) > 0 and type(v[0]) is int:  # pylint: disable=unidiomatic-typecheck
            return np.array(v, dtype=np.int64)
        return np.array(v)
    if v is None:
        # Treated as a static-optional value.
        # Dynamic optional None not yet supported.
        return v
    if isinstance(v, np.ndarray):
        return v
    raise TypeError(
        f"Unexpected onnxscript value type '{type(v)}'."
        "Valid value types are 'Tensor | list[Tensor] | None | np.ndarray | list[np.ndarray]'"
    )


def _numpy_to_onnxscript_value(
    v: np.ndarray | np.generic | list[np.ndarray] | list[np.generic],
):
    """Converts an ORT encoding of an ONNX value into the encoding used by onnxscript."""
    if isinstance(v, np.ndarray):
        return tensor.Tensor(v)
    if np.issctype(type(v)):  # noqa: NPY201
        # Numpy scalar types that are not ndarray
        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        return tensor.Tensor(np.array(v))
    if isinstance(v, list):
        return [_numpy_to_onnxscript_value(x) for x in v]
    if v is None:
        raise TypeError("Dynamic optional values not yet supported.")
    raise TypeError(
        f"Unexpected runtime value type '{type(v)}'."
        "Valid types are: 'np.ndarray | np.generic | list[np.ndarray] | list[np.generic]'"
    )


def _prepare_model_and_inputs_for_eager(
    schema: onnx.defs.OpSchema,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    implicit_args: Optional[Mapping[str, Any]],
):
    implicit_args = implicit_args or {}
    # Convert input values to ORT representation-type:
    args = [_onnxscript_to_numpy_value(x) for x in args]
    implicit_args = {k: _onnxscript_to_numpy_value(v) for k, v in implicit_args.items()}

    # Utility to convert kwarg to ONNX AttributeProto:
    def make_attr(key: str, value: Any) -> onnx.AttributeProto:
        def make_tensor_name() -> str:
            return f"attr_{key}"

        return autocast.pyvalue_to_onnx_attribute(
            key, value, make_tensor_name, schema.attributes[key].type
        )

    # Construct ONNX model with a single op call:
    inputs = [_rename_io("input", i, arg) for i, arg in enumerate(args)]

    num_outputs = compute_num_outputs(schema, args, kwargs)
    outputs = [f"output{i}" for i in range(num_outputs)]

    node = onnx.helper.make_node(schema.name, inputs, outputs, domain=schema.domain)
    node.attribute.extend(
        make_attr(key, value) for key, value in kwargs.items() if value is not None
    )
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

    session_run_input = {name: arg for name, arg in zip(inputs, args) if name != ""}
    session_run_input.update(implicit_args)

    return model, session_run_input, inputs


def _call_ort(
    schema: onnx.defs.OpSchema,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    implicit_args: Optional[Mapping[str, Any]],
):
    # Delay import onnxruntime so that onnxscript can be used without
    # installing onnxruntime.
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=import-outside-toplevel
        Fail,
        InvalidArgument,
        InvalidGraph,
    )

    model, session_run_input, inputs = _prepare_model_and_inputs_for_eager(
        schema, args, kwargs, implicit_args
    )

    try:
        session = ort.InferenceSession(
            model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
    except (Fail, InvalidGraph, InvalidArgument) as e:
        raise EagerModeError(
            f"Unable to create onnxruntime InferenceSession "
            f"for executing {schema.domain}.{schema.name} op "
            f"with onnx model\n{onnx.printer.to_text(model)}"
        ) from e

    try:
        result = session.run(None, session_run_input)
    except (RuntimeError, Fail) as e:
        raise EagerModeError(
            f"Unable to execute model operator {schema.name!r} due to {e!r}"
            f"\ninput types:\n"
            f"{pprint.pformat({k: type(v) for k, v in zip(inputs, args)})}"
            f"\nmodified input types:\n"
            f"{pprint.pformat({k: type(v) for k, v in session_run_input.items()})}"
            f"\ninputs:\n{pprint.pformat(session_run_input)}\n{model}"
        ) from e

    # Map ORT output values to the onnxscript representation-type.
    return [_numpy_to_onnxscript_value(x) for x in result]


def _schema_id(schema: onnx.defs.OpSchema) -> tuple[str, str, int]:
    return schema.name, schema.domain, schema.since_version


class ORTEvaluator(BaseEvaluator):
    """Evaluates ONNX ops using ONNX Runtime."""

    def _eval(self, schema, inputs, attributes, closure):
        return _call_ort(schema, inputs, attributes, closure)


class OnnxReferenceRuntimeEvaluator(BaseEvaluator):
    """Evaluates ONNX ops using ONNX Runtime."""

    def _eval(self, schema, inputs, attributes, closure):
        model, session_run_input, adapted_inputs = _prepare_model_and_inputs_for_eager(
            schema, inputs, attributes, closure
        )
        session = onnx.reference.ReferenceEvaluator(model)
        try:
            result = session.run(None, session_run_input)
        except RuntimeError as e:
            raise EagerModeError(
                f"Unable to execute model operator {schema.name!r} due to {e!r}"
                f"\ninput types:\n"
                f"{pprint.pformat({k: type(v) for k, v in zip(adapted_inputs, inputs)})}"
                f"\nmodified input types:\n"
                f"{pprint.pformat({k: type(v) for k, v in session_run_input.items()})}"
                f"\ninputs:\n{pprint.pformat(session_run_input)}\n{model}"
            ) from e

        return [_numpy_to_onnxscript_value(x) for x in result]


ort_evaluator = ORTEvaluator()


class ORTMixedEvaluator(ORTEvaluator):
    """Evaluates ONNX ops using ONNX Runtime, unless an overriding python implementation is registered.

    This is useful for higher-order ops such as Scan and SequenceMap, allowing for
    python-based debugging.
    """

    def __init__(self) -> None:
        super().__init__()
        self._python_ops: dict[tuple[str, str, int], Any] = {}

    def use_graph_attribute(self, schema: onnx.defs.OpSchema) -> bool:
        return _schema_id(schema) not in self._python_ops

    def _eval(self, schema, inputs, attributes, closure):
        schemaid = _schema_id(schema)
        if schemaid in self._python_ops:
            return self._python_ops[schemaid](inputs, attributes)
        else:
            return super()._eval(schema, inputs, attributes, closure)

    def register(self, opset: values.Opset) -> Callable[[_T], _T]:
        assert opset is not None

        def decorator(function: _T) -> _T:
            schema = opset[function.__name__]
            self._python_ops[_schema_id(schema)] = function
            return function

        return decorator


ort_mixed_evaluator = ORTMixedEvaluator()


@ort_mixed_evaluator.register(opset=onnx_opset.opset18)
def SequenceMap(inputs: Sequence[Any], attributes: Mapping[str, Any]):
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
