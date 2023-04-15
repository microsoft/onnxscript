# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import logging
import types
from enum import IntFlag
from typing import Any, Optional, Sequence, _GenericAlias  # type: ignore[attr-defined]

import onnx
import onnx.defs

from onnxscript import irbuilder, sourceinfo

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


class Opset:
    """Represents an ONNX Opset, which consists of a domain name, a version.

    It also contains a set of operations. This represents an Opset defined
    in the ONNX schema registry and the operations are retrieved from the
    ONNX schema registry. It also stores function definitions created for
    ops in the corresponding Opset.

    Only a single instance of Opset is created for a given (domain, version) pair.
    """

    cache: dict[tuple[type, str, int], Opset] = {}

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
            return onnx.defs.get_schema(opname, self.version, self.domain)
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return None

    def __contains__(self, opname):
        try:
            onnx.defs.get_schema(opname, self.version, self.domain)
            return True
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return False

    def __str__(self) -> str:
        return self.domain

    def __getattr__(self, attr: str):
        try:
            schema = onnx.defs.get_schema(attr, self.version, self.domain)
            return Op(self, attr, schema)
        except Exception as exc:
            raise AttributeError(f"Attribute {attr} not found.") from exc

    def add_function_def(self, fun):
        if fun.name in self.function_defs:
            logger = logging.getLogger("onnxscript")
            logger.warning("%s: Already defined.", fun.name)
        self.function_defs[fun.name] = fun

    def _prepare_inputs(self, _: onnx.defs.OpSchema, *inputs):
        """Trims 'None' values from the end of the inputs list. This is used to support
        omitting optional inputs when no more required inputs follow to prepare a valid call
        against the Op. Used by the static opset code generator.
        """
        # TODO: validate the op schema as 'None' values are removed?
        input_list = list(inputs)
        while input_list and input_list[-1] is None:
            del input_list[-1]
        return input_list


# ONNX ops


@dataclasses.dataclass(frozen=True)
class ParamSchema:
    """A schema for a parameter of an Op or a OnnxFunction.

    Attributes:
        name: The name of the parameter.
        type: The type of the parameter.
        default: The default value of the parameter.
        required: Whether the input or attribute is required.
            For example, `Slice` has two optional inputs `axes` and `steps`.
            `SoftmaxCrossEntropyLoss` has an optional attribute `ignore_index`.
        is_input: Whether the parameter is an ONNX input.
        is_variadic_input: Whether the parameter, which has to be an INPUT, is variadic.
    """

    name: str
    type: Any = None  # Op input does not have a type, for now
    default: Any = _EmptyDefault
    required: bool = True
    is_input: bool = True
    is_variadic_input: bool = False

    def __str__(self) -> str:
        """Return a string representation of the parameter.

        E.g. "x: Input[INT64]" or "axis: Attribute[int] = 0"
        """
        param_kind = "Input" if self.is_input else "Attribute"
        text = f"{self.name}: {param_kind}[{self.type}]"
        if self.default is not _EmptyDefault:
            text += f" = {self.default}"
        return text

    @property
    def is_attribute(self) -> bool:
        """Returns True if the parameter is an ONNX attribute."""
        return not self.is_input


def _get_attribute_value(attr_proto: onnx.AttributeProto) -> Any:
    """Get the default value of an ONNX attribute."""
    if attr_proto.type == onnx.AttributeProto.UNDEFINED:
        return _EmptyDefault
    return onnx.helper.get_attribute_value(attr_proto)


class Op:
    """Represents an ONNX op instance (for example, the MatMul op from ONNX opset version 13).
    It belongs to a particular Opset and has a name.

    Attributes:
        opset: The Opset that this op belongs to.
        opname: The name of the op.
        opschema: The ONNX OpSchema for the op.
    """

    def __init__(
        self, opset, opname: str, opschema: Optional[onnx.defs.OpSchema] = None
    ) -> None:
        self.opset = opset
        self.opname = opname
        self.opschema = opschema
        self._param_schemas: Optional[tuple[ParamSchema, ...]] = None

    def __call__(self, *args, **kwargs):
        # FIXME(after #225): Move import to the top of the file.
        from onnxscript import evaluator  # pylint: disable=import-outside-toplevel

        return evaluator.default().eval(self.get_schema(), args, kwargs)

    def is_single_op(self) -> bool:
        return isinstance(self.opname, str)

    def get_schema(self) -> Optional[onnx.defs.OpSchema]:
        """Returns the ONNX OpSchema for this op."""
        if self.opschema:
            return self.opschema
        return self.opset[self.opname]

    def has_schema(self) -> bool:
        """Returns True if this op has an OpSchema."""
        return self.get_schema() is not None

    def param_schemas(self) -> Optional[tuple[ParamSchema, ...]]:
        """Returns the parameter schemas for this op, if it has one."""
        if self._param_schemas is not None:
            return self._param_schemas

        op_schema = self.get_schema()
        if op_schema is None:
            return None
        schemas = []
        for input_ in op_schema.inputs:
            param_schema = ParamSchema(
                name=input_.name,
                is_input=True,
                required=(input_.option != onnx.defs.OpSchema.FormalParameterOption.Optional),
                is_variadic_input=(
                    input_.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
                ),
            )
            schemas.append(param_schema)
        for attr_name, attribute in op_schema.attributes.items():
            default_attr_proto = attribute.default_value
            param_schema = ParamSchema(
                name=attr_name,
                type=_ATTRIBUTE_TYPE_TO_PYTHON_TYPE[attribute.type],
                default=_get_attribute_value(default_attr_proto),
                is_input=False,
                required=attribute.required,
            )
            schemas.append(param_schema)

        self._param_schemas = tuple(schemas)
        return self._param_schemas  # type: ignore[return-value]


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


class OnnxFunction(Op):
    """Represents an ONNX op for which a function-body has been defined in onnxscript.

    Args:
        opset: opset the function belongs to
        pyfun: python function
        irfun: python code parsed by class
            :class:`onnxscript.converter.Converter`
        source: source code used to generate the function
        kwargs: additional properties used to construct a ModelProto
    """

    def __init__(
        self,
        opset: Opset,
        pyfun: types.FunctionType,
        irfun: irbuilder.IRFunction,
        source: str,
        kwargs: dict[str, Any],
    ):
        opset = opset or Opset(irfun.domain, 1)
        super().__init__(opset, irfun.name)
        self.function = pyfun
        self.function_ir = irfun
        self.source = source
        self.kwargs = kwargs
        self._param_schemas: Optional[tuple[ParamSchema, ...]] = None

    @property
    def name(self):
        """Returns the function name."""
        return self.opname

    def __getitem__(self, instance):
        """Returns a lambda to evaluate function using given evaluator instance.

        Usage:
            script_fun(X) executes the function using the default evaluator instance.
            script_fun[instance](X) executes the function using the given evaluator instance.
        """

        def fun(*args, **kwargs):
            # FIXME(after #225): Move import to the top of the file.
            from onnxscript import evaluator  # pylint: disable=import-outside-toplevel

            with evaluator.default_as(instance):
                return self.__call__(*args, **kwargs)

        return fun

    def __call__(self, *args, **kwargs):
        """Implements an eager-mode execution of an onnxscript function."""
        # FIXME(after #225): Move import to the top of the file.
        from onnxscript import evaluator  # pylint: disable=import-outside-toplevel

        return evaluator.default().eval_function(self, args, kwargs)

    def param_schemas(self) -> tuple[ParamSchema, ...]:
        """Returns the parameter schemas of this function."""
        if self._param_schemas is not None:
            return self._param_schemas

        function_ir = self.function_ir
        # The first len(func_ir.inputs) arguments are onnx inputs
        inputs = function_ir.inputs
        # The rest is onnx attributes

        schemas = []
        for arg in inputs:
            if isinstance(arg.typeinfo, onnx.TypeProto.Optional):
                required = False
            else:
                required = True
            schemas.append(
                ParamSchema(name=arg.name, type=arg.typeinfo, is_input=True, required=required)
            )

        for attr_parameter in function_ir.attrs:
            schemas.append(
                ParamSchema(
                    name=attr_parameter.name,
                    type=_ATTRIBUTE_TYPE_TO_PYTHON_TYPE.get(
                        onnx.defs.OpSchema.AttrType(attr_parameter.type)  # type: ignore[call-arg]
                    ),
                    default=_EmptyDefault
                    if attr_parameter.default_value is None
                    else attr_parameter.default_value,
                    is_input=False,
                    required=not attr_parameter.has_default,
                )
            )

        self._param_schemas = tuple(schemas)
        return self._param_schemas  # type: ignore[return-value]

    def to_function_proto(self):
        """Converts the function into :class:`onnx.FunctionProto`."""
        return self.function_ir.to_function_proto()

    def to_model_proto(self, **kwargs):
        """Converts the function into :class:`onnx.ModelProto`."""
        if self.function_ir.attrs and any(
            not attr.has_default for attr in self.function_ir.attrs
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
        return self.function_ir.to_model_proto(**merged_kw_args)


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

    def __init__(self, info: sourceinfo.SourceInfo) -> None:
        if not isinstance(info, sourceinfo.SourceInfo):
            raise TypeError(f"info must be of type sourceinfo.SourceInfo not {type(info)!r}.")
        self.info = info


class AttrRef(SymbolValue):
    def __init__(
        self, attr_name: str, typeinfo: _GenericAlias, info: sourceinfo.SourceInfo
    ) -> None:
        """Initializes AttrRef.

        Arguments:
            attr_name: name of the attribute-parameter
            typeinfo: type annotation of the attribute.
                op's attributes in ONNX are usually single type or list of single type.
            info: for debugging use.
        """
        super().__init__(info)
        self.value = attr_name
        self.typeinfo = typeinfo
        if not isinstance(typeinfo, (type, _GenericAlias)):
            # typing._GenericAlias for List[int] and List[str], etc.
            raise TypeError(f"Expecting a type not f{type(typeinfo)} for typeinfo.")
        self.typeinfo = typeinfo


class DynamicKind(IntFlag):
    Unknown = 0
    Input = 1
    Output = 2
    Intermediate = 4
    Loop = 8


class Dynamic(SymbolValue):
    def __init__(
        self, onnx_var: str, kind: DynamicKind, info: sourceinfo.SourceInfo, typeinfo=None
    ) -> None:
        """Initializes Dynamic.

        Arguments:
            onnx_var: the name of the ONNX variable used to represent this value
            kind: the DynamicKind of this variable
            info: source-location information for error-messages/debugging
            typeinfo: type-information for the value
        """
        super().__init__(info)
        assert isinstance(kind, DynamicKind)
        self.value = onnx_var
        self.kind = kind
        self.typeinfo = typeinfo
