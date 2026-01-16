# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ruff: noqa: TID251

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import types
import typing
from collections.abc import Collection
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

import onnx
import onnx.defs
import onnx_ir as ir
from typing_extensions import ParamSpec

from onnxscript._internal import ast_utils, irbuilder, sourceinfo
from onnxscript._internal import converter as converter_module
from onnxscript.ir import _schemas
from onnxscript.onnx_types import ONNXType

_R = TypeVar("_R")
_P = ParamSpec("_P")


def select_ir_version(version: int, domain: str = "") -> int:
    """Selects a suitable ONNX ir_version for a given opset version."""
    if domain == "":
        domain = "ai.onnx"
    if (domain, version) not in onnx.helper.OP_SET_ID_VERSION_MAP:
        return max(
            v for k, v in onnx.helper.OP_SET_ID_VERSION_MAP.items() if k[0] == "ai.onnx"
        )
    required_min_version = onnx.helper.OP_SET_ID_VERSION_MAP[domain, version]
    return max(required_min_version, 10)


_ATTRIBUTE_TYPE_TO_PYTHON_TYPE = {
    onnx.defs.OpSchema.AttrType.FLOAT: float,
    onnx.defs.OpSchema.AttrType.INT: int,
    onnx.defs.OpSchema.AttrType.STRING: str,
    onnx.defs.OpSchema.AttrType.TENSOR: None,
    onnx.defs.OpSchema.AttrType.GRAPH: None,
    onnx.defs.OpSchema.AttrType.SPARSE_TENSOR: None,
    onnx.defs.OpSchema.AttrType.TYPE_PROTO: None,
    onnx.defs.OpSchema.AttrType.FLOATS: Sequence[float],
    onnx.defs.OpSchema.AttrType.INTS: Sequence[int],
    onnx.defs.OpSchema.AttrType.STRINGS: Sequence[str],
    onnx.defs.OpSchema.AttrType.TENSORS: None,
    onnx.defs.OpSchema.AttrType.GRAPHS: None,
    onnx.defs.OpSchema.AttrType.SPARSE_TENSORS: None,
    onnx.defs.OpSchema.AttrType.TYPE_PROTOS: None,
}

# A special value to indicate that the default value is not specified
_EmptyDefault = object()

logger = logging.getLogger(__name__)


class Opset:
    """Represents an ONNX Opset, which consists of a domain name, a version.

    It also contains a set of operations. This represents an Opset defined
    in the ONNX schema registry and the operations are retrieved from the
    ONNX schema registry. It also stores function definitions created for
    ops in the corresponding Opset.

    Only a single instance of Opset is created for a given (domain, version) pair.
    """

    domain: str
    version: int
    cache: ClassVar[dict[tuple[type, str, int], Opset]] = {}

    def __new__(cls, domain: str, version: int):
        key = (cls, domain, version)
        existing = cls.cache.get(key)
        if existing:
            return existing
        instance = super().__new__(cls)
        instance.domain = domain  # type: ignore[attr-defined]
        instance.version = version  # type: ignore[attr-defined]
        instance.function_defs = {}  # type: ignore[attr-defined]
        cls.cache[key] = instance
        return instance

    def __init__(self, domain: Optional[str] = None, version: Optional[int] = None):
        # Nothing to do. Object is initialized by __new__
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.domain!r}, {self.version!r})"

    def __getitem__(self, opname):
        try:
            schema = onnx.defs.get_schema(opname, self.version, self.domain)
            return Op(self, opname, schema)
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return None

    def __contains__(self, opname):
        try:
            onnx.defs.get_schema(opname, self.version, self.domain)
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return False
        else:
            return True

    def __str__(self) -> str:
        return self.domain

    def __getattr__(self, attr: str) -> Op:
        try:
            schema = onnx.defs.get_schema(attr, self.version, self.domain)
            return Op(self, attr, schema)
        except Exception as exc:
            raise AttributeError(f"Attribute {attr} not found.") from exc

    def add_function_def(self, fun):
        self.function_defs[fun.name] = fun

    def _prepare_inputs(self, _: onnx.defs.OpSchema, *inputs):
        """Trims 'None' values from the end of the inputs list. This is used to support
        omitting optional inputs when no more required inputs follow to prepare a valid call
        against the Op. Used by the static opset code generator.
        """
        # TODO: validate the op schema as 'None' values are removed?
        input_list = list(inputs)
        while input_list and input_list[-1] is None:
            input_list.pop()
        return input_list


# ONNX ops


def _get_attribute_value(attr_proto: onnx.AttributeProto) -> Any:
    """Get the default value of an ONNX attribute."""
    if attr_proto.type == onnx.AttributeProto.UNDEFINED:
        return _EmptyDefault
    return onnx.helper.get_attribute_value(attr_proto)


def _typeinfo(var: ir.Value) -> Any:
    return var.meta.get("typeinfo")


@typing.runtime_checkable
class OpLike(Protocol):
    """A protocol for objects that have an ONNX OpSchema."""

    @property
    def name(self) -> str: ...

    @property
    def opset(self) -> Opset: ...

    @property
    def op_signature(self) -> Optional[_schemas.OpSignature]: ...


class Op(OpLike):
    """Represents an ONNX op instance (for example, the MatMul op from ONNX opset version 13).

    It belongs to a particular Opset and has a name.

    Attributes:
        opset: The Opset that this op belongs to.
        name: The name of the op.
        op_schema: The ONNX OpSchema for the op.
    """

    def __init__(
        self, opset: Opset, name: str, op_schema: Optional[onnx.defs.OpSchema] = None
    ) -> None:
        self._opset = opset
        self._name = name
        self._op_schema: onnx.defs.OpSchema | None
        if op_schema is not None:
            self._op_schema = op_schema
        elif (op := opset[name]) is not None:
            self._op_schema = op.op_schema
        else:
            self._op_schema = None
        self._signature: Optional[_schemas.OpSignature] = None

        if self._op_schema is None:
            logger.debug(
                "An OpSchema was not provided for Op '%s' and "
                "there is not one found in opset '%s'.",
                name,
                opset,
            )

    def __call__(self, *args, **kwargs):
        from onnxscript._internal import evaluator  # pylint: disable=import-outside-toplevel

        default_evaluator = evaluator.default()
        if hasattr(default_evaluator, "eval"):
            # Interface prior to onnxscript 0.6, used by PyTorch 2.10 and older
            if self.op_schema is None:
                raise ValueError(f"OpSchema not found for op '{self.name}'.")
            return default_evaluator.eval(self.op_schema, args, kwargs)
        # Use the new interface
        return evaluator.default().eval_op(self, args, kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def opset(self) -> Opset:
        return self._opset

    @property
    def op_schema(self) -> Optional[onnx.defs.OpSchema]:
        return self._op_schema

    @property
    def op_signature(self) -> Optional[_schemas.OpSignature]:
        """Returns the signature of this op."""
        if self._signature is not None:
            return self._signature

        if self.op_schema is None:
            return None

        self._signature = _schemas.OpSignature.from_op_schema(self.op_schema)
        return self._signature

    @op_signature.setter
    def op_signature(self, value: _schemas.OpSignature):
        self._signature = value


@dataclasses.dataclass(repr=False, eq=False)
class OnnxClosure:
    """Represents a nested function used as a graph-valued attribute for an ONNX op call."""

    function_ir: irbuilder.IRFunction

    # frame is python's stack-frame for the execution of top-level
    # script function (in eager-mode). It is used to get the current
    # value of outer-scope variables referred to inside this nested
    # function/GraphProto.
    frame: types.FrameType

    function: Any


class OnnxFunction(Op, Generic[_P, _R]):
    """Represents an ONNX op for which a function-body has been defined in onnxscript.

    Attributes:
        opset: Opset the function belongs to.
        name: Name of the function.
        function: Python function.
        function_ir: Python code parsed as an :class:`irbuilder.IRFunction`.
        source: Source code used to generate the function.
        kwargs: Additional properties used to construct a ModelProto.
        op_schema: Generated ONNX OpSchema for this op.
    """

    def __init__(
        self,
        opset: Optional[Opset],
        pyfun: Callable,
        irfun: irbuilder.IRFunction,
        source: str,
        kwargs: dict[str, Any],
    ):
        """Constructs an OnnxFunction.

        Args:
            opset: opset the function belongs to
            pyfun: python function
            irfun: python code parsed by class
                :class:`onnxscript.converter.Converter`
            source: source code used to generate the function
            kwargs: additional properties used to construct a ModelProto
        """
        opset = opset or Opset(irfun.domain, 1)
        super().__init__(opset, irfun.name)
        self.function = pyfun
        self.function_ir = irfun
        self.source = source
        self.kwargs = kwargs
        self._signature = _schemas.OpSignature.from_function(
            self.function, domain=self.function_ir.domain, name=self.name
        )

        # Allow the object to be inspected as a function
        functools.update_wrapper(self, pyfun)

        # Experimental fields
        self.traceable = False

    @property
    def op_signature(self) -> Optional[_schemas.OpSignature]:
        """Returns the signature of this op."""
        return self._signature

    @op_signature.setter
    def op_signature(self, value: _schemas.OpSignature):
        self._signature = value

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Implements an eager-mode execution of an onnxscript function."""
        from onnxscript._internal import evaluator  # pylint: disable=import-outside-toplevel

        return evaluator.default().eval_function(self, args, kwargs)  # type: ignore[arg-type, return-value]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.function!r})"

    def to_function_proto(self) -> onnx.FunctionProto:
        """Converts the function into :class:`onnx.FunctionProto`."""
        return self.function_ir.to_function_proto()

    def to_model_proto(self, **kwargs):
        """Converts the function into :class:`onnx.ModelProto`."""
        if self.function_ir.attrs and any(
            attr.value is None for attr in self.function_ir.attrs
        ):
            raise ValueError(
                "A function with required attributes cannot be exported as a model."
            )
        # Note: The function must also have monomorphic type annotation for inputs/outputs
        # to be converted into a valid model. Otherwise, we can still produce an ONNX
        # model, but it will not pass the ONNX model checker. We do not report an error
        # at this stage.

        # Merge kwargs specified in script-decorator with those specified in this call.
        merged_kw_args = {**self.kwargs, **kwargs}
        return self._to_model_proto(**merged_kw_args)

    def _to_model_proto(
        self,
        functions: Collection[ir.Function] | None = None,
        io_types: Optional[ONNXType] = None,
        input_types: Optional[Sequence[ONNXType]] = None,
        output_types: Optional[Sequence[ONNXType]] = None,
        value_infos: dict[str, ONNXType] | None = None,
        opset_version: int | None = None,
        **kwargs,
    ) -> onnx.ModelProto:
        """Converts this instance into a `onnx.ModelProto`.

        Args:
            functions: A list of functions to include in the model.
                By default, all functions called at least once are included.
            io_types: When specified, all the inputs/outputs of the model
                are set to be of this type.
            input_types: When specified, all the inputs of the model
                are set to be of the corresponding type in this list.
            output_types: When specified, all the outputs of the model
                are set to be of the corresponding type in this list.
            value_infos: A dictionary mapping intermediate variable names to ONNX types.
                Used to set value_info for intermediate variables.
            opset_version: The standard opset version to use for the model if it
                cannot be inferred. Otherwise defaults to the current opset version.
            kwargs: Additional parameters given to function :func:`onnx.helper.make_model`.

        Returns:
            An instance of :class:`onnx.ModelProto`.
        """
        # Identify functions to include in the model
        if functions is None:
            sub_functions = self.function_ir.get_called_functions()
            functions = sub_functions.values()

        # Determine opset imports
        opsets = self.function_ir.graph.opset_imports

        for func in functions:
            if func.domain not in opsets:
                opsets[func.domain] = 1

        # No need to collect opsets from functions

        # FIXME: Collect used opsets from the function nodes
        if "" not in opsets:
            # No operator is using the standard opset.
            # Use the specified version if provided or the default value.
            opsets[""] = (
                opset_version if opset_version is not None else onnx.defs.onnx_opset_version()
            )

        # Determine ir_version
        if "ir_version" in kwargs:
            ir_version = kwargs.pop("ir_version")
        else:
            ir_version = select_ir_version(opsets[""])

        # Create the model
        model = ir.Model(self.function_ir.graph, ir_version=ir_version)
        for func in functions:
            model.functions[func.identifier()] = func

        model_proto = ir.to_proto(model)

        # Set additional type information if provided
        graph = model_proto.graph

        if value_infos:
            graph.value_info.extend(
                [
                    onnx.helper.make_value_info(name, type.to_type_proto())
                    for name, type in value_infos.items()
                ]
            )

        if io_types is not None:
            for input in graph.input:
                if not input.HasField("type"):
                    input.type.CopyFrom(io_types.to_type_proto())
            for output in graph.output:
                if not output.HasField("type"):
                    output.type.CopyFrom(io_types.to_type_proto())
        if input_types is not None:
            for input, type in zip(graph.input, input_types):
                input.type.CopyFrom(type.to_type_proto())
        if output_types is not None:
            for output, type in zip(graph.output, output_types):
                output.type.CopyFrom(type.to_type_proto())

        for k, v in kwargs.items():
            setattr(model_proto, k, v)

        return model_proto


class TracedOnnxFunction(Op):
    """TracedOnnxFunction.

    Attributes:
        name: Name of the op. E.g. "aten::add".
        func: Function.
    """

    def __init__(self, opset: Opset, func: Callable):
        super().__init__(opset, func.__name__)
        self.func = func
        self._signature = _schemas.OpSignature.from_function(
            self.func, domain="_traced", name=self.name
        )

        # Allow the object to be inspected as a function
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.func!r})"

    @property
    def function_ir(self) -> irbuilder.IRFunction:
        """Return the function_ir.

        This function IR contains only the signature of the function.
        """
        src, func_ast = ast_utils.get_src_and_ast(self.func)
        module = inspect.getmodule(self.func)
        closure = inspect.getclosurevars(self.func)
        global_names = module.__dict__.copy()
        global_names.update(closure.nonlocals)
        converter = converter_module.Converter(
            opset=self._opset,
            global_names=global_names,
            source=src,
        )

        return converter.translate_function_signature(func_ast)

    @property
    def op_signature(self) -> Optional[_schemas.OpSignature]:
        """Returns the signature of this op."""
        return self._signature

    @op_signature.setter
    def op_signature(self, value: _schemas.OpSignature):
        self._signature = value


class SymbolValue:
    """Represents script-time value information about named variables used in a script.

    At translation-time, the (local) variables of a script, including its parameters,
    are bound to a SymbolValue.

    SymbolValues fall into the following categories:

    AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes

    Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs.
    Dynamic values include input-parameters of the script, as well intermediate
    values computed in the script.

    For example, consider the following script definition:
    ::

        @script()
        def ThresholdedRelu(X, alpha: float):
            zero = op.CastLike(0, X)
            return op.Where(X > alpha, X, zero)

    Here, `X` has a Dynamic value, `alpha` has an AttrRef value, and `zero`
    has a Dynamic value.

    Scripts may also contain references to global variables, but the translator
    does not associate a SymbolValue with them. The python value of global variables
    is used directly in the translation, and such global variables are intended
    to be used for limited purposes, namely:
    * To identify an opset
    * To represent constant-values, translated into ONNX constants.
    """

    def __init__(self, value: Any, info: sourceinfo.SourceInfo) -> None:
        """
        Initializes SymbolValue.

        Arguments:
            value: The value bound to a python variable in a script.
            info: source-location information for error-messages/debugging
        """
        if not isinstance(info, sourceinfo.SourceInfo):
            raise TypeError(f"info must be of type sourceinfo.SourceInfo not {type(info)!r}.")
        self.value = value
        self.info = info


class AttrRef(SymbolValue):
    def __init__(self, attr: ir.Attr, as_bool: bool, info: sourceinfo.SourceInfo) -> None:
        """Initializes AttrRef.

        Arguments:
            attr: An ir.Attr representing the attribute-parameter
            as_bool: Whether the attribute is to be interpreted as a bool type (represented as int in ONNX)
            info: for debugging use.
        """
        super().__init__(attr, info)
        self.as_bool = as_bool
