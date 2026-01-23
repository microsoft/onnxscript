# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, Literal, Sequence, cast, overload

import libcst as cst
import libcst.matchers as cstm
import libcst.metadata as cstmeta
import onnx

from onnxscript.codeanalysis import (
    CstCodeGenerator,
    RemoveUnusedImportsTransformer,
    format_code,
    make_const_expr,
)

__all__ = [
    "OnnxScriptCodeGenerator",
    "OnnxScriptTransformer",
    "OnnxToPythonOperatorTransformer",
    "OnnxConstantOpToPythonConstantTransformer",
    "Driver",
]

DEFAULT_OPSET_VERSION: Final = 18


@dataclass
class QualifiedOnnxOp:
    domain: str
    name: str
    version: int = 0

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.domain == "" and self.name == __value
        elif isinstance(__value, QualifiedOnnxOp):
            return (
                self.domain == __value.domain
                and self.name == __value.name
                and (__value.version <= 0 or self.version == __value.version)
            )
        return False


class OnnxScriptCodeGenerator(CstCodeGenerator):
    def __init__(self):
        super().__init__()
        self.__opset_version = DEFAULT_OPSET_VERSION

    def translate_model_proto(self, model_proto: onnx.ModelProto) -> cst.Module:
        for opset_import in model_proto.opset_import:
            if opset_import.domain == "":
                self.__opset_version = opset_import.version

        self.add_import_from("__future__", "annotations")

        model_local_functions = [
            self.translate_function_proto(func) for func in model_proto.functions
        ]
        model_main_function = self.__make_function(model_proto.graph, "script")

        return cst.Module(
            body=[
                *self.make_import_statements(),
                model_main_function,
                *model_local_functions,
            ]
        )

    def translate_function_proto(self, function_proto: onnx.FunctionProto) -> cst.FunctionDef:
        return self.__make_function(function_proto, "script")

    def translate_graph_proto(
        self,
        function_proto: onnx.GraphProto,
        func_type: Literal["graph"] | Literal["script"] = "graph",
    ) -> cst.FunctionDef:
        return self.__make_function(function_proto, func_type)

    def translate_tensor_proto(self, tensor_proto: onnx.TensorProto) -> cst.BaseExpression:
        if onnx.external_data_helper.uses_external_data(tensor_proto):
            raise NotImplementedError("tensors with external data are not supported")

        self.add_import("onnx")

        numpy_tensor = onnx.numpy_helper.to_array(tensor_proto)

        return cst.Call(
            func=cst.Attribute(
                cst.Attribute(cst.Name("onnx"), cst.Name("helper")),
                cst.Name("make_tensor"),
            ),
            args=[
                cst.Arg(
                    cst.SimpleString('"value"'),
                    keyword=cst.Name("name"),
                ),
                cst.Arg(
                    self.__make_onnx_dtype_expr(tensor_proto.data_type),
                    keyword=cst.Name("data_type"),
                ),
                cst.Arg(
                    cst.List(
                        elements=[
                            cst.Element(make_const_expr(dim)) for dim in numpy_tensor.shape
                        ]
                    ),
                    keyword=cst.Name("dims"),
                ),
                cst.Arg(
                    cst.List(
                        elements=[
                            cst.Element(make_const_expr(val))
                            for val in numpy_tensor.ravel().tolist()
                        ],
                    ),
                    keyword=cst.Name("vals"),
                ),
            ],
        )

    def translate_type_proto(self, type_proto: onnx.TypeProto) -> cst.BaseExpression | None:
        if type_proto.WhichOneof("value") == "tensor_type":
            return self.translate_tensor_type_proto(type_proto.tensor_type)

        return None

    def translate_tensor_type_proto(
        self, tensor_type: onnx.TypeProto.Tensor
    ) -> cst.Name | cst.Subscript:
        pytype_name = onnx.helper.tensor_dtype_to_string(tensor_type.elem_type)
        tensor_proto_prefix = "TensorProto."
        if not pytype_name.startswith(tensor_proto_prefix):
            raise NotImplementedError(pytype_name)
        pytype_name = pytype_name[len(tensor_proto_prefix) :]

        self.add_import_from("onnxscript", pytype_name)
        pytype = cst.Name(pytype_name)

        if not tensor_type.HasField("shape"):
            # unknown shape, e.g. FLOAT[...]
            return cst.Subscript(
                value=pytype, slice=[cst.SubscriptElement(cst.Index(cst.Ellipsis()))]
            )

        if len(tensor_type.shape.dim):
            # have dimensions, e.g. FLOAT[2, 3] or FLOAT["M", "N"] and so on
            return cst.Subscript(
                value=pytype,
                slice=[
                    cst.SubscriptElement(slice=cst.Index(value=self.translate_dimension(dim)))
                    for dim in tensor_type.shape.dim
                ],
            )

        # scalar, e.g. FLOAT
        return pytype

    def translate_dimension(
        self, dimension: onnx.TensorShapeProto.Dimension
    ) -> cst.Integer | cst.SimpleString:
        kind = dimension.WhichOneof("value")
        if kind == "dim_value":
            return cst.Integer(str(dimension.dim_value))
        elif kind == "dim_param":
            return cst.SimpleString(f'"{dimension.dim_param}"')
        raise NotImplementedError(kind)

    def translate_node_proto(self, node_proto: onnx.NodeProto):
        for attr in node_proto.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield self.__make_function(attr.g, "graph")
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield self.__make_function(g, "graph")

        op_call = cst.Call(
            func=self.__make_op_name(node_proto),
            args=[cst.Arg(cst.Name(input)) for input in node_proto.input]
            + [
                cst.Arg(self.__make_attr_value(attr), keyword=cst.Name(attr.name))
                for attr in node_proto.attribute
            ],
        )

        yield cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[
                        cst.AssignTarget(cst.Name(output)) for output in node_proto.output
                    ],
                    value=op_call,
                )
            ]
        )

    def __make_function(
        self,
        proto: onnx.GraphProto | onnx.FunctionProto,
        func_type: Literal["graph"] | Literal["script"],
    ) -> cst.FunctionDef:
        params: list[cst.Param] = []

        self.add_import_from("onnxscript", func_type)

        def maybe_annotation(expr: cst.BaseExpression | None) -> cst.Annotation | None:
            return cst.Annotation(expr) if expr is not None else None

        if isinstance(proto, onnx.FunctionProto):
            params = [cst.Param(cst.Name(input)) for input in proto.input]
            returns = [cst.Name(output) for output in proto.output]
        else:
            params = [
                cst.Param(
                    name=cst.Name(input.name),
                    annotation=maybe_annotation(self.translate_type_proto(input.type)),
                )
                for input in proto.input
            ]
            returns = [cst.Name(output.name) for output in proto.output]

        body: list[cst.BaseStatement] = []

        for node in proto.node:
            for stmt in self.translate_node_proto(node):
                body.append(stmt)

        assert len(returns) > 0
        body.append(
            cst.SimpleStatementLine(
                body=[
                    cst.Return(
                        cst.Tuple(elements=[cst.Element(ret) for ret in returns])
                        if len(returns) > 1
                        else returns[0]
                    )
                ]
            )
        )

        return cst.FunctionDef(
            name=cst.Name(proto.name),
            params=cst.Parameters(params),
            body=cst.IndentedBlock(
                body=body,
            ),
            decorators=[cst.Decorator(cst.Call(func=cst.Name(func_type)))],
        )

    def __make_op_name(self, node_proto: onnx.NodeProto):
        opset = node_proto.domain
        if not opset:
            opset = "op"
            self.add_import_from(
                module="onnxscript",
                name=f"opset{self.__opset_version}",
                alias=opset,
            )
        return cst.Attribute(value=cst.Name(opset), attr=cst.Name(node_proto.op_type))

    def __make_attr_value(self, attr: onnx.AttributeProto):
        if attr.type == onnx.AttributeProto.INT:
            return make_const_expr(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            return cst.List(
                elements=[cst.Element(make_const_expr(i)) for i in attr.ints],
            )
        elif attr.type == onnx.AttributeProto.FLOAT:
            return make_const_expr(attr.f)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return cst.List(
                elements=[cst.Element(make_const_expr(f)) for f in attr.floats],
            )
        elif attr.type == onnx.AttributeProto.TENSOR:
            return self.translate_tensor_proto(attr.t)
        elif attr.type == onnx.AttributeProto.GRAPH:
            return cst.Name(attr.g.name)

        raise NotImplementedError(f"attr.type={attr.type}, attr: {attr}")

    def __make_onnx_dtype_expr(self, dtype: int):
        dtype_name = onnx.helper.tensor_dtype_to_string(dtype).split(".")[-1]
        return cst.Attribute(
            cst.Attribute(cst.Name("onnx"), cst.Name("TensorProto")),
            cst.Name(dtype_name),
        )


class OnnxScriptTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cstmeta.QualifiedNameProvider,)

    @overload
    def matches_qualified_name(
        self,
        node: cst.CSTNode,
        qualname: str,
        source: cstmeta.QualifiedNameSource | None = None,
    ) -> bool:
        ...

    @overload
    def matches_qualified_name(
        self,
        node: cst.CSTNode,
        qualname: re.Pattern[str],
        source: cstmeta.QualifiedNameSource | None = None,
    ) -> re.Match[str] | None:
        ...

    def matches_qualified_name(
        self,
        node: cst.CSTNode,
        qualname: str | re.Pattern[str],
        source: cstmeta.QualifiedNameSource | None = None,
    ) -> re.Match[str] | bool | None:
        for resolved_qualname in self.get_metadata(cstmeta.QualifiedNameProvider, node, set()):
            match = (
                qualname.match(resolved_qualname.name)
                if isinstance(qualname, re.Pattern)
                else resolved_qualname.name == qualname
            )
            if match and (source is None or source == resolved_qualname.source):
                return match
        return None if isinstance(qualname, re.Pattern) else False

    def resolve_onnx_op(self, node: cst.CSTNode) -> QualifiedOnnxOp | None:
        if match := self.matches_qualified_name(
            node,
            re.compile(r"^onnxscript\.opset(\d+)\.(\w+)$"),
            cstmeta.QualifiedNameSource.IMPORT,
        ):
            # TODO: we need to parse and analyze the entire import to resolve
            # the qualified name to ensure it's bound to a FunctionDef whose
            # parent is a ClassDef with that a base of onnxscript.values.Opset.
            opset_version, op_name = match.groups()
            return QualifiedOnnxOp(domain="", name=op_name, version=int(opset_version))
        return None


class OnnxToPythonOperatorTransformer(OnnxScriptTransformer):
    def __init__(self):
        super().__init__()
        self.__transforms: Final[dict[str, type[cst.CSTNode]]] = {
            # Binary Operators
            "Add": cst.Add,
            "Sub": cst.Subtract,
            "Mul": cst.Multiply,
            "MatMul": cst.MatrixMultiply,
            "Div": cst.Divide,
            "Pow": cst.Power,
            "BitwiseAnd": cst.BitAnd,
            "BitwiseOr": cst.BitOr,
            "BitwiseXor": cst.BitXor,
            # Boolean Operators
            "And": cst.And,
            "Or": cst.Or,
        }

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if (
            (onnx_op := self.resolve_onnx_op(original_node)) is None
            or onnx_op.domain != ""
            or (pynode_type := self.__transforms.get(onnx_op.name)) is None
        ):
            return updated_node

        pynode = pynode_type()

        if (is_binary := isinstance(pynode, cst.BaseBinaryOp)) or isinstance(
            pynode, cst.BaseBooleanOp
        ):
            assert len(updated_node.args) == 2
            return (cst.BinaryOperation if is_binary else cst.BooleanOperation)(
                left=updated_node.args[0].value,
                operator=pynode,
                right=updated_node.args[1].value,
            )
        elif isinstance(pynode, cst.BaseUnaryOp):
            assert len(updated_node.args) == 1
            return cst.UnaryOperation(
                operator=pynode,
                expression=updated_node.args[0].value,
            )

        raise NotImplementedError(pynode)


class OnnxConstantOpToPythonConstantTransformer(OnnxScriptTransformer):
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        # Constant(...) must have exactly one value kwarg
        if self.resolve_onnx_op(original_node) != "Constant" or not cstm.matches(
            original_node,
            cstm.Call(
                args=[
                    cstm.Arg(
                        keyword=cstm.Name("value")
                        | cstm.Name("value_float")
                        | cstm.Name("value_floats")
                        | cstm.Name("value_int")
                        | cstm.Name("value_ints")
                        | cstm.Name("value_string")
                        | cstm.Name("value_strings")
                    )
                ]
            ),
        ):
            return updated_node

        def get_kwarg_and_value_expr(node: cst.Call):
            return cast(cst.Name, node.args[0].keyword).value, node.args[0].value

        kwarg, value_expr = get_kwarg_and_value_expr(original_node)

        # Constant(value=make_tensor(...))
        if (
            kwarg == "value"
            and isinstance(value_expr, cst.Call)
            and self.matches_qualified_name(value_expr, "onnx.helper.make_tensor")
            and 4 <= len(value_expr.args) <= 5
        ):
            updated_node = self.__transform_value_make_tensor(updated_node, value_expr)
            kwarg, value_expr = get_kwarg_and_value_expr(updated_node)
            if kwarg == "value":
                # call could not be transformed to one of the simpler forms, so bail
                return updated_node

        # Constant(value_(int|float|string)s?=...)
        return value_expr

    def __transform_value_make_tensor(
        self, constant_op_call: cst.Call, constant_op_arg_expr: cst.Call
    ) -> cst.Call:
        # deduce dtype from an integer constant or an 'onnx.TensorProto.<DTYPE>' expr
        dtype = onnx.TensorProto.UNDEFINED
        dtype_expr = constant_op_arg_expr.args[1]
        if isinstance(dtype_expr.value, cst.Integer):
            dtype = cast(onnx.TensorProto.DataType, int(dtype_expr.value.value))
        elif (
            dtype_match := self.matches_qualified_name(
                dtype_expr.value, re.compile(r"onnx.TensorProto.(\w+)")
            )
        ) and hasattr(onnx.TensorProto, dtype_match.group(1)):
            dtype = getattr(onnx.TensorProto, dtype_match.group(1))
        if dtype not in (
            onnx.TensorProto.INT64,
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.STRING,
        ):
            return constant_op_call

        # support len(dims) of 0 (scalar) or 1 (list) for rewriting to
        # value_T or value_Ts, respectively
        dims_expr = constant_op_arg_expr.args[2].value
        if not isinstance(dims_expr, (cst.List, cst.Tuple)) or len(dims_expr.elements) > 1:
            return constant_op_call

        # vals must be a list or tuple whose elements are integer or float constants
        vals_expr = constant_op_arg_expr.args[3].value
        if not isinstance(vals_expr, (cst.List, cst.Tuple)) or not all(
            isinstance(e.value, (cst.Integer, cst.Float, cst.SimpleString))
            for e in vals_expr.elements
        ):
            return constant_op_call

        if dtype == onnx.TensorProto.INT64:
            kwarg = "value_int"
        elif dtype == onnx.TensorProto.FLOAT:
            kwarg = "value_float"
        elif dtype == onnx.TensorProto.STRING:
            kwarg = "value_string"

        if len(dims_expr.elements) == 0:
            if len(vals_expr.elements) != 1:
                return constant_op_call
            vals_expr = vals_expr.elements[0].value
        else:
            kwarg += "s"

        return constant_op_call.with_changes(
            args=[cst.Arg(value=vals_expr, keyword=cst.Name(kwarg))]
        )


class Driver:
    DEFAULT_TRANSFORMER_TYPES: Final[Sequence[type[cst.CSTTransformer]]] = [
        OnnxConstantOpToPythonConstantTransformer,
        OnnxToPythonOperatorTransformer,
        RemoveUnusedImportsTransformer,
    ]

    def __init__(
        self,
        model: onnx.ModelProto | Path | str | BinaryIO,
        transformers: Sequence[cst.CSTTransformer] | None = None,
    ):
        if isinstance(model, Path):
            model = str(model.resolve())
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load_model(model)

        self.model: Final = model
        self.__transformers: Final = (
            [t() for t in Driver.DEFAULT_TRANSFORMER_TYPES]
            if transformers is None
            else list(transformers)
        )

    @property
    def transformers(self) -> Sequence[cst.CSTTransformer]:
        return self.__transformers

    def to_cst_module(self) -> cst.Module:
        codegen = OnnxScriptCodeGenerator()
        cst_module = codegen.translate_model_proto(self.model)
        cst_module = codegen.apply_transformers(cst_module, self.transformers)
        return cst_module

    def to_python_code(self, reference_path: Path | None = None) -> bytes:
        return format_code(
            path=reference_path,
            code=self.to_cst_module().bytes,
        )
