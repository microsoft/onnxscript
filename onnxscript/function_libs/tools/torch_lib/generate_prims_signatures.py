# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generates the Prims signatures for the ONNX Prims operator set using torch.ops."""
from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Sequence

import torch
import torchgen.gen
import torchgen.model
from torch._ops import _OpNamespace
from torchgen.model import FunctionSchema

import opgen.pygen as cg


def create_list_type(arg: torchgen.model.Argument) -> cg.TypeRef:
    inner_arg_type = arg.type if not arg.type.is_nullable() else arg.type.elem
    assert isinstance(inner_arg_type, torchgen.model.ListType), f"arg: {arg}"

    arg_type = arg_type_to_str(arg.type)
    if type_is_builtin(arg_type):
        return cg.TypingRefs.Sequence(cg.BuiltinTypeRef(arg_type))
    if arg_type == "TensorType":
        return cg.TypingRefs.Sequence(cg.TypeRef("onnxscript.onnx_types", "TensorType"))
    return cg.TypeRef("onnxscript", arg_type)

    # TODO(justinchuby): Enable this when generics are better supported
    # if arg.type.size is None:
    #     # INT64[...]
    #     return cg.TypeRef("onnxscript", arg_type, cg.EllipsisTypeRef())
    # # INT64[3]
    # return cg.TypeRef("onnxscript", arg_type, *[cg.TypeRef(None, f"{arg.type.size}")])


def arg_type_to_str(arg_type: torchgen.model.Type) -> str:
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.Tensor):
        return "TensorType"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.SymInt):
        return "INT64"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.Scalar):
        return "float"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.float):
        return "float"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.int):
        return "int"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.bool):
        return "bool"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.str):
        return "str"
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.ScalarType):
        return "int"

    # Anything unhandled is a string option.
    return "str"


def type_is_builtin(arg_type: str) -> bool:
    """Returns whether the given type is a python builtin type (that we care about)."""
    return arg_type in {"float", "int", "bool", "str"}


def get_argument_type(arg: torchgen.model.Argument) -> cg.TypeRef:
    """Returns the Python type for the given argument."""
    inner_arg_type = arg.type if not arg.type.is_nullable() else arg.type.elem

    if isinstance(inner_arg_type, torchgen.model.ListType):
        inner_node = create_list_type(arg)
    else:
        arg_type_str = arg_type_to_str(inner_arg_type)
        if type_is_builtin(arg_type_str):
            inner_node = cg.BuiltinTypeRef(arg_type_str)
        elif arg_type_str == "TensorType":
            inner_node = cg.TypeRef("onnxscript.onnx_types", "TensorType")
        else:
            inner_node = cg.TypeRef("onnxscript", arg_type_str)

    if arg.type.is_nullable():
        return cg.TypingRefs.Optional(inner_node)
    if arg.default is not None and parse_default_value(arg) is None:
        return cg.TypingRefs.Optional(inner_node)
    return inner_node


def should_generate_signature(op_name: str, schema: FunctionSchema) -> bool:
    """Returns whether the signature for the given function should be generated."""
    if op_name.split("_")[1].startswith("_"):
        return False
    if schema.name.name.inplace:
        return False
    if schema.name.overload_name and schema.name.overload_name != "Tensor":
        # Ignore overloads for now.
        # Some ops only have overloaded versions, like aten::add.Tensor. And we
        # want to generate the aten::add op.
        return False
    return True


def parse_default_value(arg: torchgen.model.Argument) -> Any:
    default = arg.default
    assert default is not None, f"arg: {arg}"
    if default.startswith("[") and default.endswith("]"):
        # Convert list to tuple
        default_val = ast.literal_eval(default)
        assert isinstance(default_val, list)
        if not default_val:
            # Empty list is represented as None.
            return None
        return tuple(default_val)
    # Special case for reduction=Mean
    if default == "Mean":
        return 1

    try:
        value = ast.literal_eval(default)
        if isinstance(value, int):
            # Expand the value to a tuple if the type is a list.
            if isinstance(arg.type, torchgen.model.ListType):
                if arg.type.size is not None:
                    return (value,) * arg.type.size
                return (value,)
        return value
    except ValueError:
        # Treat it as a string.
        return default.lower()


def create_return_type(returns: Sequence[torchgen.model.Return]) -> cg.TypeRef:
    """Returns the Python type for the return value of the given function."""
    if not returns:
        return cg.TypingRefs.Any()
    return_nodes = []
    for return_val in returns:
        return_type = return_val.type
        return_type_str = arg_type_to_str(return_type)
        if type_is_builtin(return_type_str):
            # Python type
            return_node: cg.TypeRef = cg.BuiltinTypeRef(return_type_str)
        elif return_type_str == "TensorType":
            return_node = cg.TypeRef("onnxscript.onnx_types", "TensorType")
        else:
            return_node = cg.TypeRef("onnxscript", arg_type_to_str(return_type))
        if return_type.is_nullable():
            return_node = cg.TypingRefs.Optional(return_node)
        return_nodes.append(return_node)
    if len(return_nodes) == 1:
        return return_nodes[0]
    return cg.BuiltinTypeRef("tuple", *return_nodes)


def format_arg_name(arg: torchgen.model.Argument) -> str:
    """Returns the python compatible name of the given argument."""
    if arg.name == "from":
        return f"{arg.name}_"
    return arg.name


def create_signature(op_name: str, schema: FunctionSchema) -> cg.FunctionDef:
    """Creates the signature for the given function."""
    args = [
        arg.argument if isinstance(arg, torchgen.model.SelfArgument) else arg
        for arg in schema.arguments.positional
    ]
    kwargs = [
        arg
        for arg in schema.arguments.kwarg_only
        if not isinstance(arg, torchgen.model.TensorOptionsArguments)
    ]

    py_args = [
        cg.Arg(
            format_arg_name(arg),
            get_argument_type(arg),
            default_value=cg.Constant(parse_default_value(arg))
            if arg.default is not None
            else None,
        )
        for arg in args
    ]
    if kwargs:
        # Arguments after this point are keyword-only.
        py_args += [
            cg.Arg(
                format_arg_name(kwarg),
                get_argument_type(kwarg),
                default_value=cg.Constant(parse_default_value(kwarg))
                if kwarg.default is not None
                else None,
                is_kwarg=True,
            )
            for kwarg in kwargs
        ]

    return cg.FunctionDef(
        op_name,
        *py_args,
        return_type=create_return_type(schema.returns),
        body=[
            cg.ThunkStmt(f'"""{schema}"""'),
            cg.Raise(cg.Call(cg.Name("NotImplementedError"))),  # type: ignore[list-item]
        ],
    )


def create_onnx_function_module(
    functions: Sequence[tuple(str, FunctionSchema)],  # type: ignore[valid-type]
) -> cg.Module:
    """Creates the onnx function module."""
    return cg.Module(
        cg.ImportFrom("__future__", cg.Alias("annotations")),
        *[
            create_signature(name, schema)
            for name, schema in functions
            if should_generate_signature(name, schema)
        ],
    )


def copyright_header() -> str:
    """Creates the copyright header."""
    dashline = f"# {'-' * 74}"
    return textwrap.dedent(
        f"""\
        {dashline}
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License.
        {dashline}
        # mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
        """
    )


def _get_func_schema_in_namespace(namespaces: List[_OpNamespace]) -> Dict[str, FunctionSchema]:
    table: Dict[str, FunctionSchema] = {}
    for op_namespace in namespaces:
        for attr_name in dir(op_namespace):
            op_overload_packet = getattr(op_namespace, attr_name)
            if not isinstance(
                op_overload_packet,
                torch._ops.OpOverloadPacket,  # pylint: disable=protected-access
            ):
                continue

            # Update schema to avoid returning mutable positional args
            # which fails FunctionSchema.parse(). For example:
            # change "resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)"
            # to "resize(Tensor a, SymInt[] shape) -> Tensor"
            if "!" in op_overload_packet.schema:
                op_overload_packet.schema = re.sub(  # type: ignore[attr-defined]
                    "[(][A-Za-z]![)]", "", op_overload_packet.schema
                )

            # FIXME: remove below code if the issue below is fixed.
            # https://github.com/pytorch/pytorch/issues/99714
            op_overload_packet.schema = op_overload_packet.schema.replace(",  ", ", ")  # type: ignore[attr-defined]

            func_schema = FunctionSchema.parse(op_overload_packet.schema)
            op_name = op_namespace.name + "_" + attr_name
            table[op_name] = func_schema

    return table


def main(args: argparse.Namespace) -> None:
    all_functions = _get_func_schema_in_namespace([torch.ops.prims])
    functions: dict[str, dict[str, FunctionSchema]] = {}
    for op_name, func_schema in all_functions.items():
        if not should_generate_signature(op_name, func_schema):
            continue

        module_name = op_name.split("_")[0]
        if module_name not in functions:
            functions[module_name] = {}
        if op_name in functions[module_name]:
            logging.warning(
                "Duplicated function: %s, overload: %s",
                op_name,
                func_schema.name.overload_name,
            )
            continue
        functions[module_name][op_name] = func_schema

    os.makedirs(args.outdir, exist_ok=True)

    for module_name, module_functions in functions.items():
        sorted_functions = sorted(module_functions.items(), key=lambda x: x[0])
        py_module = create_onnx_function_module(list(sorted_functions))
        py_module.accept(cg.ImportAdjuster())
        py_module.accept(cg.DocCommentBuilder())
        output_path = os.path.join(args.outdir, f"{module_name}.py")

        print(f"Generating {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(copyright_header())
            # Add docstring
            f.write(
                textwrap.dedent(
                    f'''\
                    """torch.ops.aten operators under the `{module_name}` module.

                    - No inplace operators.
                    - All functions should not have the script() decorator. This is because
                        we want to delay the compilation of the function.
                    """
                '''
                )
            )
            py_module.accept(cg.PythonWriter(f))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory for generated modules",
    )
    main(parser.parse_args())
