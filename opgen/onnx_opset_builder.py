# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import os
import pathlib
import textwrap
from typing import Any, Iterable, Optional, TextIO

import onnx.defs
import onnx.helper

from onnxscript._internal.pygen import pygen as cg

__all__ = [
    "QualOpName",
    "UnsupportedOpError",
    "OpsetsBuilder",
    "parse_attr_type",
    "parse_input_output_type",
]

MODULE_ONNX = "onnx"
MODULE_ONNX_DEFS = "onnx.defs"
MODULE_ONNX_SCRIPT_TYPES = "onnxscript.onnx_types"
MODULE_ONNX_SCRIPT_VALUES = "onnxscript.values"


class QualOpName:
    def __init__(self, domain: str, name: str, version: int):
        self.domain = domain
        self.name = name
        self.version = version
        self.docuri = (
            "https://onnx.ai/onnx/operators/onnx_"
            f"{domain.replace('.', '')}_{name}.html#{name.lower()}-{version}"
        )

    def __repr__(self) -> str:
        return (
            f"QualOpName(domain={self.domain!r}, "
            f"version={self.version!r}, name={self.name!r})"
        )

    def __str__(self) -> str:
        domain_prefix = f"{self.domain}::" if self.domain else ""
        return f"{domain_prefix}{self.name}({self.version})"


class OpsetBaseTypeRef(cg.TypeRef):
    def __init__(self):
        super().__init__(MODULE_ONNX_SCRIPT_VALUES, "Opset")


class TensorTypeRef(cg.TypeRef):
    def __init__(self):
        super().__init__(MODULE_ONNX_SCRIPT_TYPES, "Tensor")


class UnsupportedOpError(NotImplementedError):
    def __init__(self, op: QualOpName, message: str):
        super().__init__(self, message)
        self.op = op
        self.message = message


def _make_suffix(str: str) -> str:
    return f"_{str.replace('.', '_')}" if str else ""


def _make_class_name(domain: str, version: int) -> str:
    return f"Opset{_make_suffix(domain)}{version}"


def _make_module_name(base_name: str, domain: str, version: int) -> str:
    return f"{base_name}._impl.opset{_make_suffix(domain)}{version}"


class OpsetModule(cg.Module):
    def __init__(self, base_name: str, domain: str, version: int, *members: cg.Stmt):
        self.domain = domain
        self.version = version
        super().__init__(*members, name=_make_module_name(base_name, domain, version))


class OpsetsBuilder:
    def __init__(self, module_base_name: str, min_default_opset_version: int):
        self.module_base_name = module_base_name
        self.min_default_opset_version = min_default_opset_version
        self.all_ops_count: int = 0
        self.all_modules: list[cg.Module] = []
        self.unsupported_ops: dict[str, list[UnsupportedOpError]] = {}
        self._make_opset_modules()
        self._make_init_module()
        self._make_imports()

    def _log_unsupported(self, error: UnsupportedOpError):
        self.unsupported_ops.setdefault(error.message, []).append(error)

    def _make_opset_module(self, domain: str, version: int):
        if version > 1:
            base_type = cg.TypeRef(
                _make_module_name(self.module_base_name, domain, version - 1),
                _make_class_name(domain, version - 1),
            )
        else:
            base_type = OpsetBaseTypeRef()

        opset = OpsetModule(
            self.module_base_name,
            domain,
            version,
            cg.ClassDef(
                _make_class_name(domain, version),
                cg.FunctionDef(
                    "__new__",
                    cg.Arg("cls"),
                    body=cg.ThunkStmt(
                        f"return Opset.__new__(cls, " f"{domain!r}, {version!r})"
                    ),
                ),
                cg.FunctionDef(
                    "__init__", cg.Arg("self"), body=cg.ThunkStmt("super().__init__()")
                ),
                bases=[base_type],
            ),
        )

        self.all_modules.append(opset)
        return opset

    def _make_opset_modules(self):
        domains = {}
        schemas: list[onnx.defs.OpSchema] = sorted(
            onnx.defs.get_all_schemas_with_history(),
            key=lambda op: (op.domain, op.since_version, op.name),
        )

        for schema in schemas:
            qualname = QualOpName(schema.domain, schema.name, schema.since_version)
            domain: str = schema.domain
            version: int = schema.since_version
            domain_opsets = domains.setdefault(domain, {})

            if schema.deprecated:
                self._log_unsupported(UnsupportedOpError(qualname, "deprecated"))
                continue

            if version in domain_opsets:
                opset = domain_opsets[version]
            else:
                opset = self._make_opset_module(domain, version)
                domain_opsets[version] = opset

            try:
                function = self._make_function(qualname, schema)
                opset_class = cg.first_or_none(opset.get_children_of_type(cg.ClassDef))
                if opset_class:
                    opset_class.append_body(function)
                    self.all_ops_count += 1
            except NotImplementedError as error:
                if not isinstance(error, UnsupportedOpError):
                    error = UnsupportedOpError(qualname, str(error))
                self._log_unsupported(error)

        if onnx.defs.onnx_opset_version() not in domains[""]:
            self._make_opset_module("", onnx.defs.onnx_opset_version())

        for module in self.all_modules:
            module.accept(cg.DocCommentBuilder())

        self.all_modules.sort(key=lambda m: (m.domain, m.version, m.name))

    def _make_init_module(self):
        all_list = cg.ListExpr(cg.Constant("default_opset"), cg.Constant("all_opsets"))
        init_module = cg.Module(
            cg.ImportFrom(MODULE_ONNX_DEFS, cg.Alias("onnx_opset_version")),
            cg.Assign(cg.Name("__all__"), all_list),
            cg.If(
                cg.BinOp(
                    cg.Call(cg.Name("onnx_opset_version")),
                    "<",
                    cg.Constant(self.min_default_opset_version),
                ),
                cg.Raise(
                    cg.Call(
                        cg.Name("ImportError"),
                        cg.ThunkExpr(
                            'f"ONNX Script requires ONNX opset >= '
                            f"{self.min_default_opset_version} "
                            'but {onnx_opset_version()} is detected."'
                        ),
                    )
                ),
            ),
            name=f"{self.module_base_name}.__init__",
        )

        all_opsets = cg.DictExpr()
        for opset_module in filter(lambda m: isinstance(m, OpsetModule), self.all_modules):
            opset_module: OpsetModule
            opset_class = cg.first_or_none(opset_module.get_children_of_type(cg.ClassDef))
            if opset_class is not None:
                opset_export_name = opset_module.name.split(".")[-1]
                all_opsets.append_element(
                    cg.DictElem(
                        cg.TupleExpr(
                            cg.Constant(opset_module.domain), cg.Constant(opset_module.version)
                        ),
                        cg.Name(opset_export_name),
                    )
                )
                all_list.append_child(
                    cg.Constant(opset_export_name), cg.ListExpr.Roles.Elements
                )
                init_module.append_body(
                    cg.Assign(cg.Name(opset_export_name), cg.Call(opset_class.make_typeref()))
                )
        all_opsets_type = cg.TypeRef.make_composite_if_multiple(
            cg.TypingRefs.Mapping,
            cg.TypeRef.make_composite_if_multiple(
                cg.TypingRefs.Tuple, cg.StrTypeRef(), cg.IntTypeRef()
            ),
            cg.TypeRef(MODULE_ONNX_SCRIPT_VALUES, "Opset"),
        )
        init_module.append_body(cg.Assign(cg.Name("all_opsets"), all_opsets, all_opsets_type))

        default_opset = cg.Assign(
            cg.Name("default_opset"),
            cg.Subscript(
                cg.Name("all_opsets"),
                cg.TupleExpr(cg.Constant(""), cg.Call(cg.Name("onnx_opset_version"))),
            ),
            cg.TypeRef(None, f"Opset{self.min_default_opset_version}"),
        )
        default_opset.trailing_trivia = "  # type: ignore"
        init_module.append_body(default_opset)

        self.all_modules.append(init_module)

    def _make_imports(self):
        for module in self.all_modules:
            if isinstance(module, OpsetModule):
                module.prepend_child(
                    cg.ImportFrom(MODULE_ONNX_DEFS, cg.Alias("get_schema")),
                    cg.Module.Roles.Body,
                )
                module.prepend_child(
                    cg.ImportFrom(MODULE_ONNX_SCRIPT_VALUES, cg.Alias("Op, Opset")),
                    cg.Module.Roles.Body,
                )
            module.accept(cg.ImportAdjuster())

    def _make_function(
        self, qualname: QualOpName, schema: onnx.defs.OpSchema
    ) -> cg.FunctionDef:
        op_inputs: list[cg.Expr] = []
        op_attrs: list[cg.Expr] = []
        args = list(self._make_function_args(schema))

        for arg in args:
            if arg.name == "self":
                continue
            if arg.is_vararg:
                op_inputs.append(cg.Starred(cg.Name(arg.name)))
            elif arg.is_kwarg:
                op_attrs.append(cg.Assign(cg.Name(arg.name), cg.Name(arg.name)))
            else:
                op_inputs.append(cg.Name(arg.name))

        if len(op_inputs) > 0:
            op_call = cg.Call(
                cg.Name("op"),
                cg.Starred(
                    cg.Call(cg.Name("self._prepare_inputs"), cg.Name("schema"), *op_inputs)
                ),
                *op_attrs,
            )
        else:
            op_call = cg.Call(cg.Name("op"), *op_attrs)

        doc = f'[🌐 {qualname}]({qualname.docuri} "Online Documentation")\n\n{schema.doc}'

        def return_type():
            return cg.TypeRef.make_composite_if_multiple(
                cg.TypingRefs.Tuple,
                *[self._make_union_typeref(output.types) for output in schema.outputs],
            )

        func = cg.FunctionDef(
            qualname.name,
            *args,
            return_type=return_type(),
            doc=_process_documentation(doc),
            body=[
                cg.Assign(
                    cg.Name("schema"),
                    cg.Call(
                        cg.Name("get_schema"),
                        cg.Constant(qualname.name),
                        cg.Constant(qualname.version),
                        cg.Constant(qualname.domain),
                    ),
                ),
                cg.Assign(
                    cg.Name("op"),
                    cg.Call(
                        cg.Name("Op"),
                        cg.Name("self"),
                        cg.Constant(qualname.name),
                        cg.Name("schema"),
                    ),
                    cg.TypingRefs.Callable(cg.EllipsisTypeRef(), return_type()),
                ),
                cg.Return(op_call),
            ],
        )

        return func

    def _make_function_args(self, schema: onnx.defs.OpSchema) -> Iterable[cg.Arg]:
        yield cg.Arg("self")
        yield from self._make_function_input_args(schema)
        yield from self._make_function_attr_args(schema)

    def _make_input_arg_name(self, input_name: str, schema: onnx.defs.OpSchema):
        """ONNX allows for an op to have an input and an attribute with the same name.
        Attribute names have contextual meaning however, so detect this case and disambiguate
        the input name. See Split(1) for the only offending onnx.defs.OpSchema as of opset 18.
        """
        for attr in schema.attributes.values():
            if attr.name == input_name:
                return f"{input_name}_"
        return input_name

    def _make_function_input_args(self, schema: onnx.defs.OpSchema) -> Iterable[cg.Arg]:
        args: list[cg.Arg] = []
        for input in schema.inputs:
            optional = input.option == onnx.defs.OpSchema.FormalParameterOption.Optional
            variadic = input.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
            heterogeneous = not input.isHomogeneous
            differentiable = (
                input.differentiationCategory
                == onnx.defs.OpSchema.DifferentiationCategory.Differentiable
            )
            non_differentiable = (
                input.differentiationCategory
                == onnx.defs.OpSchema.DifferentiationCategory.NonDifferentiable
            )

            doctags = []
            if optional:
                doctags.append("optional")
            elif variadic:
                # if we encounter a variadic input, previous
                # inputs cannot have default values
                for prev_arg in args:
                    prev_arg.default_value = None
                doctags.append("variadic")
            if heterogeneous:
                doctags.append("heterogeneous")
            if differentiable:
                doctags.append("differentiable")
            elif non_differentiable:
                doctags.append("non-differentiable")

            doc = input.description.strip()
            if len(doctags) > 0:
                doc = f"({', '.join(doctags)}) {doc}"

            type = self._make_union_typeref(input.types)
            if optional and not isinstance(type, cg.TypingRefs.Optional):
                type = cg.TypingRefs.Optional(type)

            args.append(
                cg.Arg(
                    self._make_input_arg_name(input.name, schema),
                    type=type,
                    doc=_process_documentation(doc),
                    is_vararg=variadic,
                    default_value=cg.Constant(None) if optional else None,
                )
            )

        return args

    def _make_function_attr_args(self, schema: onnx.defs.OpSchema) -> Iterable[cg.Arg]:
        attr_args = []
        for attr in schema.attributes.values():
            attr_type = parse_attr_type(attr.type)
            default_value = None

            if attr.required:
                pass
            elif attr.default_value.name:
                default_value = onnx.helper.get_attribute_value(attr.default_value)

                def fmt(value: Any) -> str:
                    if isinstance(value, (bytes, bytearray)):
                        return str(value.decode("utf-8"))
                    return value

                if isinstance(default_value, list):
                    default_value = tuple(fmt(val) for val in default_value)
                else:
                    default_value = fmt(default_value)
            else:
                default_value = None

            if default_value is None:
                attr_type = cg.TypingRefs.Optional(attr_type)

            attr_args.append(
                cg.Arg(
                    attr.name,
                    type=attr_type,
                    default_value=cg.Constant(default_value),
                    doc=attr.description,
                    is_kwarg=True,
                )
            )

        yield from sorted(attr_args, key=lambda p: p.has_default_value)

    def _make_union_typeref(self, onnx_types: list[str]) -> cg.TypingRefs.Union:
        return cg.TypeRef.make_composite_if_multiple(
            cg.TypingRefs.Union,
            *[parse_input_output_type(type) for type in sorted(onnx_types)],
        )

    def write(self, base_path: pathlib.Path) -> list[pathlib.Path]:
        return sorted([self._write_module(base_path, module) for module in self.all_modules])

    def _write_module(self, base_path: pathlib.Path, module: cg.Module) -> pathlib.Path:
        qual_name = module.name.split(".")
        base_path = base_path.joinpath(*qual_name[:-1])
        os.makedirs(base_path, exist_ok=True)
        path = base_path.joinpath(qual_name[-1] + ".py")
        with open(path, "w", encoding="utf-8") as writer:
            self._write_header(writer)
            module.accept(cg.PythonWriter(writer))
        return path

    def _write_header(self, writer: TextIO):
        dashline = f"# {'-' * 74}\n"
        writer.write(dashline)
        writer.write("# ⚠️ WARNING - AUTO-GENERATED CODE - DO NOT EDIT ⚠️ \n")
        writer.write("# ⚙️ Generated by 'python -m opgen'\n")
        writer.write(dashline)
        writer.write("# Copyright (c) Microsoft Corporation. ")
        writer.write("All rights reserved.\n")
        writer.write("# Licensed under the MIT License.\n")
        writer.write(dashline)
        writer.write("# flake8: noqa\n")
        writer.write("# mypy: disable-error-code=override\n")
        writer.write("# pylint: disable=W0221,W0222,W0237,W0246,R0901,W0611\n")
        writer.write(dashline)
        writer.write("\n")


def parse_input_output_type(onnx_type: str) -> cg.TypeRef:
    def error(message: Optional[str] = None):
        return NotImplementedError(
            f"input/output type not implemented: {onnx_type!r}"
            + (f" ({message!r})" if message else "")
        )

    default_value_map = {
        "BOOL": bool(),
        "FLOAT": float(),
        "FLOAT16": float(),
        "BFLOAT16": float(),
        "DOUBLE": float(),
        "INT8": int(),
        "INT16": int(),
        "INT32": int(),
        "INT64": int(),
        "UINT8": int(),
        "UINT16": int(),
        "UINT32": int(),
        "UINT64": int(),
        "COMPLEX64": complex(),
        "COMPLEX128": complex(),
    }

    id = ""
    stack: list[cg.TypeRef] = []
    for c in onnx_type:
        if c == "(":
            if id == "tensor":
                type = TensorTypeRef()
            elif id == "seq":
                type = cg.TypingRefs.Sequence()
            elif id == "map":
                type = cg.TypingRefs.Mapping()
            elif id == "optional":
                type = cg.TypingRefs.Optional()
            else:
                raise error(id)
            if len(stack) > 0:
                stack[-1].append_typearg(type)
            stack.append(type)
            id = ""
        elif c in (")", ","):
            type = stack.pop() if c == ")" else stack[-1]
            if isinstance(type, TensorTypeRef):
                type.name = id.upper()
                type.default_value = cg.Constant(default_value_map.get(type.name))
            elif id and isinstance(type, cg.TypingRefs.Mapping):
                if id == "int64":
                    type.append_typearg(cg.IntTypeRef())
                elif id == "string":
                    type.append_typearg(cg.StrTypeRef())
                else:
                    raise error(id)
            elif id:
                break
            id = ""
            if len(stack) == 0:
                return type
        else:
            id += c
    raise error()


def parse_attr_type(type) -> cg.TypeRef:
    if type == onnx.defs.AttributeProto.FLOAT:
        return cg.FloatTypeRef()
    if type == onnx.defs.AttributeProto.INT:
        return cg.IntTypeRef()
    if type == onnx.defs.AttributeProto.STRING:
        return cg.StrTypeRef()
    if type == onnx.defs.AttributeProto.TENSOR:
        return cg.TypeRef(MODULE_ONNX, "TensorProto")
    if type == onnx.defs.AttributeProto.SPARSE_TENSOR:
        return cg.TypeRef(MODULE_ONNX, "SparseTensorProto")
    if type == onnx.defs.AttributeProto.GRAPH:
        return cg.TypeRef(MODULE_ONNX, "GraphProto")
    if type == onnx.defs.AttributeProto.TYPE_PROTO:
        return cg.TypeRef(MODULE_ONNX, "TypeProto")
    if type == onnx.defs.AttributeProto.FLOATS:
        return cg.TypingRefs.Sequence(cg.FloatTypeRef())
    if type == onnx.defs.AttributeProto.INTS:
        return cg.TypingRefs.Sequence(cg.IntTypeRef())
    if type == onnx.defs.AttributeProto.STRINGS:
        return cg.TypingRefs.Sequence(cg.StrTypeRef())
    if type == onnx.defs.AttributeProto.TENSORS:
        return cg.TypingRefs.Sequence(cg.TypeRef(MODULE_ONNX, "TensorProto"))
    if type == onnx.defs.AttributeProto.SPARSE_TENSORS:
        return cg.TypingRefs.Sequence(cg.TypeRef(MODULE_ONNX, "SparseTensorProto"))
    if type == onnx.defs.AttributeProto.GRAPHS:
        return cg.TypingRefs.Sequence(cg.TypeRef(MODULE_ONNX, "GraphProto"))
    if type == onnx.defs.AttributeProto.TYPE_PROTOS:
        return cg.TypingRefs.Sequence(cg.TypeRef(MODULE_ONNX, "TypeProto"))
    raise NotImplementedError(f"attribute type not implemented: {type}")


def _process_documentation(doc: str):
    # Lifted from ONNX's docsgen:
    # https://github.com/onnx/onnx/blob/3fd41d249bb8006935aa0031a332dd945e61b7e5/docs/docsgen/source/onnx_sphinx.py#L414
    doc = textwrap.dedent(doc or "")
    main_docs_url = "https://github.com/onnx/onnx/blob/master/"
    rep = {
        "[the doc](IR.md)": "`ONNX <{0}docs/IR.md>`_",
        "[the doc](Broadcasting.md)": "`Broadcasting in ONNX <{0}docs/Broadcasting.md>`_",
        "<dl>": "",
        "</dl>": "",
        "<dt>": "* ",
        "<dd>": "  ",
        "</dt>": "",
        "</dd>": "",
        "<tt>": "``",
        "</tt>": "``",
        "<br>": "\n",
    }
    for k, v in rep.items():
        doc = doc.replace(k, v.format(main_docs_url))
    move = 0
    lines = []
    for line in doc.split("\n"):
        if line.startswith("```"):
            if move > 0:
                move -= 4
                lines.append("\n")
            else:
                lines.append("::\n")
                move += 4
        elif move > 0:
            lines.append(" " * move + line)
        else:
            lines.append(line)
    return "\n".join(lines)
