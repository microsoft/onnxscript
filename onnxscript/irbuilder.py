# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import io
import logging
import warnings
from typing import Any, Optional, Protocol, Sequence, Union

import onnx
from onnx import ValueInfoProto, helper
from onnx.defs import onnx_opset_version

import onnxscript
from onnxscript import type_annotation as ta
from onnxscript import values
from onnxscript._internal import version_utils
from onnxscript.onnx_types import ONNXType
from onnxscript.sourceinfo import SourceInfo

# A simple IR (Function, Stmt, Attr, Var):

logger = logging.getLogger("onnxscript")


def _format(seq: Sequence[Any], prefix: str, sep: str, suffix: str, formatter=str):
    """Formats a sequence of objects into a string."""
    return prefix + sep.join([formatter(x) for x in seq]) + suffix


def select_ir_version(version: int, domain: str = ""):
    """Selects a suitable ONNX ir_version for a given opset version."""
    if domain == "":
        domain = "ai.onnx"
    if (domain, version) not in helper.OP_SET_ID_VERSION_MAP:
        return max(v for k, v in helper.OP_SET_ID_VERSION_MAP.items() if k[0] == "ai.onnx")
    return helper.OP_SET_ID_VERSION_MAP[domain, version]


class IRType:
    def __init__(self):
        self.onnx_type = onnx.TypeProto()

    def to_type_proto(self):
        return self.onnx_type

    def __repr__(self) -> str:
        return "IRType()"


class IRTensorType(IRType):
    def __init__(self, elem_type: onnx.TensorProto.DataType) -> None:
        super().__init__()
        self.onnx_type.tensor_type.elem_type = elem_type

    def __repr__(self) -> str:
        return f"IRTensorType({self.onnx_type.tensor_type.elem_type})"


class IRTypeLike(Protocol):
    def to_type_proto(self) -> onnx.TypeProto:
        """Converts IR type representation to onnx.TypeProto"""


class IRVar:
    """A variable (representing a formal parameter)."""

    def __init__(self, varname: str, typeinfo: IRTypeLike, sourceinfo: SourceInfo) -> None:
        if not isinstance(varname, str):
            raise ValueError(f"varname must be a string not {type(varname)!r}.")
        self.name = varname
        self.info = sourceinfo
        self.typeinfo = typeinfo

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.typeinfo!r})"

    def typed_str(self):
        return f"{self.name} : {self.typeinfo}"

    def to_value_info(self, use_default_type: bool = True):
        """Converts the content of this class into :class:`onnx.ValueInfoProto`.

        Args:
            use_default_type: if True, use a default type if an explicit type
                is not known. Otherwise, returns a ValueInfoProto without type.

        Returns:
            an instance of :class:`onnx.ValueInfoProto`
        """
        if self.name is None:
            raise ValueError(self.info.msg("name cannot be None."))
        value_info_proto = ValueInfoProto()
        value_info_proto.name = self.name
        if self.typeinfo is not None:
            value_info_proto.type.CopyFrom(self.typeinfo.to_type_proto())
        elif use_default_type:
            value_info_proto.type.CopyFrom(IRType().to_type_proto())
        return value_info_proto


def _opt_var_to_str(x):
    return "" if x is None else str(x)


class IRAttributeValue:
    """An attribute value (representing an actual parameter).

    Attributes:
        name: The name of the attribute.
        type: The type of the attribute.
        attr_proto: The attribute proto.
    """

    def __init__(self, attrproto: onnx.AttributeProto) -> None:
        self.attr_proto = attrproto

    def __str__(self):
        if self.attr_proto.HasField("ref_attr_name"):
            return f"{self.attr_proto.name} = @{self.attr_proto.ref_attr_name}"
        # self.name + " = " + self.value
        return helper.printable_attribute(self.attr_proto)

    @property
    def name(self) -> str:
        return self.attr_proto.name

    @property
    def type(self) -> onnx.AttributeProto.AttributeType:
        return self.attr_proto.type


@dataclasses.dataclass(frozen=True)
class IRAttributeParameter:
    """An attribute parameter (representing a formal parameter).

    It may or may not carry a default value.

    Attributes:
        name: The name of the attribute.
        type: The type of the attribute.
        default_value: The default value of the attribute.
        has_default: Whether the attribute has a default value.
        attr_proto: The attribute proto.
    """

    name: str
    type: onnx.AttributeProto.AttributeType
    default_value: str | int | float | None = None

    # TODO(justinchuby): Validate the default_value is the same type as specified in AttributeType.

    def __str__(self):
        if self.has_default:
            return helper.printable_attribute(self.attr_proto)
        # TODO(justinchuby): Include a readable type name.
        return self.name

    @property
    def has_default(self):
        return self.default_value is not None

    @property
    def attr_proto(self) -> onnx.AttributeProto:
        if not self.has_default:
            raise ValueError(
                "Attribute has no default value. Only attributes with default "
                "values can be converted to AttributeProto."
            )
        if version_utils.onnx_older_than("1.15"):
            # TODO(after 1.14 is deprecated): Remove this branch.
            # Argument 'attr_type' was added after version 1.14.
            return helper.make_attribute(self.name, self.default_value)
        # pylint: disable=unexpected-keyword-arg
        return helper.make_attribute(self.name, self.default_value, attr_type=self.type)  # type: ignore[call-arg]
        # pylint: enable=unexpected-keyword-arg


class IRStmt:
    def __init__(
        self,
        result: Sequence[str],
        callee: values.Op,
        args: Sequence[Optional[str]],
        attrs: Sequence[IRAttributeValue],
        sub_functions=None,
    ) -> None:
        if not isinstance(callee, values.Op):
            raise TypeError(f"Unexpected type {type(callee)} for callee.")
        self.result = result
        self.callee = callee
        self.args = args
        self.attrs = attrs
        self.functions = sub_functions or {}

    def __str__(self):
        if isinstance(self.result, str):
            logger.debug("unexpected str type for self.result where type(self)=%r", type(self))
        lhs = ", ".join(self.result)
        attrs = ""
        if self.attrs:
            attrs = _format(self.attrs, "<", ", ", ">")

        args = _format(self.args, "(", ", ", ")", _opt_var_to_str)
        domain = self.callee.opset.domain
        opname = self.callee.name
        callee = f"{domain}.{opname}" if (domain != "") else opname
        return f"{lhs} = {callee} {attrs}{args}"

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: %s", type(self), str(self))

    def to_node_proto(self, node_name: str) -> onnx.NodeProto:
        n = helper.make_node(
            self.callee.name,
            [_opt_var_to_str(x) for x in self.args],
            [str(x) for x in self.result],
            domain=self.callee.opset.domain,
            name=node_name,
        )
        for a in self.attrs:
            n.attribute.append(a.attr_proto)
        return n

    @property
    def output_names(self) -> Sequence[str]:
        """Returns the list of variables assigned to by this statement."""
        return [str(x) for x in self.result]


class IRFunction:
    """Represents a function in the IR."""

    def __init__(self, name: str, domain: str = "") -> None:
        self.domain = domain
        self.name = name
        self.outputs: list[IRVar] = []
        self.stmts: list[IRStmt] = []
        self.called_functions: dict[str, onnx.FunctionProto] = {}
        self.docstring: str = ""
        # a dictionary of nested function-definitions
        self.nested_functions: dict[str, IRFunction] = {}
        self.outer_scope_variables: dict[Any, Any] = {}
        self.ordered_inputs_and_attrs: list[Union[IRVar, IRAttributeParameter]] = []

    @property
    def assigned_names(self) -> Sequence[str]:
        """Returns the list of variables assigned to by this function."""
        return [v for stmt in self.stmts for v in stmt.output_names]

    @property
    def inputs(self) -> Sequence[IRVar]:
        return [var for var in self.ordered_inputs_and_attrs if isinstance(var, IRVar)]

    @property
    def attrs(self) -> Sequence[IRAttributeParameter]:
        return [
            attr
            for attr in self.ordered_inputs_and_attrs
            if isinstance(attr, IRAttributeParameter)
        ]

    def __str__(self):
        attrs = _format(self.attrs, "<", ", ", ">") if self.attrs else ""
        inputs = _format([x.typed_str() for x in self.inputs], "(", ", ", ")")
        outputs = _format([x.typed_str() for x in self.outputs], "(", ", ", ")")
        stmts = _format(self.stmts, "\n{\n   ", "\n   ", "\n}\n")
        return f"{self.name} {attrs}{inputs} => {outputs}{stmts}"

    def append_docstring(self, docstring):
        self.docstring += docstring

    def append_stmt(self, stmt: IRStmt) -> None:
        self.stmts.append(stmt)

    def append_input(self, name: IRVar) -> None:
        self.ordered_inputs_and_attrs.append(name)

    def append_output(self, name: IRVar) -> None:
        self.outputs.append(name)

    def add_attr_parameter(self, attr: IRAttributeParameter) -> None:
        self.ordered_inputs_and_attrs.append(attr)

    def debug_print(self):
        if logger.isEnabledFor(logging.DEBUG):
            st = io.StringIO()
            for s in self.stmts:
                for attr in s.attrs:
                    if attr.attr_proto.HasField("g"):
                        st.write(helper.printable_graph(attr.attr_proto.g))
                        st.write("\n")

    def add_called_function(self, fun: values.OnnxFunction) -> None:
        for name, fct in fun.function_ir.called_functions.items():
            if name in self.called_functions:
                continue
            self.called_functions[name] = fct
        if fun.name in self.called_functions:
            # Already added.
            return
        try:
            proto = fun.to_function_proto()
        except (TypeError, AttributeError) as e:
            raise TypeError(f"Issue with type f{type(fun)}.") from e
        self.called_functions[fun.name] = proto

    def add_nested_function(self, fun: IRFunction) -> None:
        self.nested_functions[fun.name] = fun

    def to_model_proto(
        self,
        functions=None,
        io_types: Optional[ONNXType] = None,
        input_types: Optional[Sequence[ONNXType]] = None,
        output_types: Optional[Sequence[ONNXType]] = None,
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
            kwargs: Additional parameters given to function :func:`onnx.helper.make_model`.

        Returns:
            An instance of :class:`onnx.ModelProto`.
        """
        graph, sub_functions = self.to_graph_and_functions(use_default_type=False)
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
        if functions is None:
            functions = sub_functions.values()
        else:

            def to_proto(f):
                if isinstance(f, onnx.FunctionProto):
                    return f
                if isinstance(f, onnxscript.OnnxFunction):
                    return f.to_function_proto()
                raise TypeError("Expected a value of type FunctionProto of OnnxFunction")

            functions = [to_proto(f) for f in functions]

        opsets = {}
        for n in self.stmts:
            if n.callee.opset.domain not in opsets:
                opsets[n.callee.opset.domain] = n.callee.opset.version
        if "" not in opsets:
            # No operator is using the standard opset.
            # A default value is given.
            opsets[""] = onnx_opset_version()
        for proto in functions:
            if proto.domain not in opsets:
                opsets[proto.domain] = 1

        if "ir_version" not in kwargs:
            kwargs["ir_version"] = select_ir_version(opsets[""])
        opset_imports = [
            onnx.helper.make_opsetid(domain, version) for domain, version in opsets.items()
        ]

        return helper.make_model(
            graph, opset_imports=opset_imports, functions=functions, **kwargs
        )

    def to_graph_and_functions(
        self, use_default_type: bool = True
    ) -> tuple[onnx.GraphProto, dict[str, onnx.FunctionProto]]:
        """Converts this instance into a `onnx.GraphProto` and a map from
        function-name to `onnx.FunctionProto`.

        Args:
            use_default_type: if True, the function uses a default type
                for inputs and outputs that do not have a type

        Returns:
            a pair of a :class:`onnx.GraphProto` and list of :class:`onnx.FunctionProto`
        """
        called_functions: dict[str, onnx.FunctionProto] = {}
        for s in self.stmts:
            called_functions.update(s.functions)
        called_functions.update(self.called_functions)
        graph = helper.make_graph(
            [s.to_node_proto(f"n{i}") for i, s in enumerate(self.stmts)],
            self.name,
            [x.to_value_info(use_default_type) for x in self.inputs],
            [y.to_value_info(use_default_type) for y in self.outputs],
        )
        return graph, called_functions

    def to_graph_proto(self, use_default_type: bool = True) -> onnx.GraphProto:
        """Converts this instance into a `onnx.GraphProto`.

        Args:
            use_default_type: if True, the function uses a default type
                for inputs and outputs that do not have a type

        Returns:
            an instance of :class:`onnx.GraphProto`
        """
        graph, _ = self.to_graph_and_functions(use_default_type=use_default_type)
        return graph

    def get_opset_import(self) -> dict[str, int]:
        func_opset_imports = {}
        for s in self.stmts:
            if s.callee.opset.domain not in func_opset_imports:
                func_opset_imports[s.callee.opset.domain] = s.callee.opset.version
            elif func_opset_imports[s.callee.opset.domain] != s.callee.opset.version:
                warnings.warn(
                    f"There is a version conflict in domain: {s.callee.opset.domain!r}, "
                    f"with {self.name!r}.",
                    category=UserWarning,
                    stacklevel=1,
                )
        return func_opset_imports

    def to_function_proto(self) -> onnx.FunctionProto:
        """Converts this instance into a `onnx.FunctionProto`.

        Note: Default values for attributes are an experimental feature in ONNX.
        Conversion ignores default values for attributes if the ONNX version installed
        doesn't support it.
        """
        opsets = self.get_opset_import()
        nodes = [s.to_node_proto(f"n{i}") for i, s in enumerate(self.stmts)]
        for n in nodes:
            if n.domain not in opsets:
                opsets[n.domain] = 1  # TODO: how to get n.version?
        opset_imports = [
            onnx.helper.make_opsetid(domain, version) for domain, version in opsets.items()
        ]

        attribute_names = [attr.name for attr in self.attrs if not attr.has_default]

        f = helper.make_function(
            self.domain,
            self.name,
            inputs=[x.name for x in self.inputs],
            outputs=[y.name for y in self.outputs],
            nodes=nodes,
            opset_imports=opset_imports,  # TODO
            attributes=attribute_names,
            doc_string=self.docstring,
        )
        # In protobuf 4.x fields aren't defined as class attribute so it should check instance attribute instead
        if hasattr(f, "attribute_proto"):
            f.attribute_proto.extend(
                [attr.attr_proto for attr in self.attrs if attr.has_default]
            )
        return f


# IRBuilder: abstracts out details of the IR in the python-to-IR converter


class IRBuilder:
    def __init__(self):
        self.functions = {}

    def new_function(self, name: str, domain: str = "", register: bool = False) -> IRFunction:
        if register and (domain, name) in self.functions:
            raise RuntimeError(f"Function '{name}' already exists in domain '{domain}'.")
        function = IRFunction(name, domain)
        if register:
            self.functions[domain, name] = function
        return function

    def add_docstring(self, fn: IRFunction, docstring: str):
        fn.append_docstring(docstring)

    def add_stmt(
        self,
        fn: IRFunction,
        results: Sequence[str],
        callee: values.Op,
        args: Sequence[Optional[str]],
        attrs: Sequence[IRAttributeValue],
        sub_functions=None,
    ) -> None:
        stmt = IRStmt(results, callee, args, attrs, sub_functions=sub_functions)
        fn.append_stmt(stmt)

    def add_input(
        self, fn: IRFunction, varname: str, type: IRTypeLike, info: SourceInfo
    ) -> None:
        var = IRVar(varname, type, info)
        fn.append_input(var)

    def add_attr_parameter(
        self,
        fn: IRFunction,
        varname: str,
        attribute_type: onnx.AttributeProto.AttributeType,
        default_value: int | float | str | None,
    ) -> None:
        fn.add_attr_parameter(IRAttributeParameter(varname, attribute_type, default_value))

    def add_output(self, fn: IRFunction, varname: str, typeinfo, sourceinfo) -> None:
        var = IRVar(varname, typeinfo, sourceinfo)
        fn.append_output(var)

    def make_attr(self, attrproto: onnx.AttributeProto) -> IRAttributeValue:
        return IRAttributeValue(attrproto)

    def make_attr_ref(self, attrname: str, refname: str, pytype: type) -> IRAttributeValue:
        proto = onnx.AttributeProto()
        proto.name = attrname
        proto.ref_attr_name = refname
        attr_type = ta.pytype_to_attrtype(pytype)
        assert attr_type is not None
        proto.type = attr_type
        return IRAttributeValue(proto)
