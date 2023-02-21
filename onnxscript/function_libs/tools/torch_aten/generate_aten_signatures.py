# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generates the ATen signatures for the ONNX ATen operator set using torch.ops."""
from __future__ import annotations

import argparse
import ast
import logging
import os
import textwrap
import typing
from typing import Any, Sequence

import torchgen.gen
import torchgen.model
import yaml

import opgen.pygen as cg


def load_native_function_yaml(yaml_path: str):
    with open(yaml_path, encoding="utf-8") as f:
        yaml_str = f.read()
    with open(yaml_path, encoding="utf-8") as f:
        all_functions = yaml.safe_load(f)
    valid_tags = set()
    # Mark all tags as valid, since we don't want to validate them.
    for func in all_functions:
        if "tags" not in func:
            continue
        valid_tags.add(func["tags"])

    return yaml_str, valid_tags


def parse_native_functions_yaml(yaml_path: str) -> tuple[Any, Any]:
    """Parses the native_functions.yaml file."""
    yaml_str, valid_tags = load_native_function_yaml(yaml_path)
    yaml_struct = yaml.load(yaml_str, Loader=torchgen.gen.LineLoader)
    parsed = torchgen.gen.parse_native_yaml_struct(
        yaml_struct, valid_tags, path=yaml_path, skip_native_fns_gen=True
    )
    return parsed.native_functions, parsed.backend_indices


def create_list_type(arg: torchgen.model.Argument) -> cg.TypeRef:
    assert isinstance(arg.type, torchgen.model.ListType), f"arg: {arg}"
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
    if isinstance(arg.type, torchgen.model.ListType):
        inner_node = create_list_type(arg)
    else:
        arg_type_str = arg_type_to_str(arg.type)
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


def should_generate_signature(func: torchgen.model.NativeFunction) -> bool:
    """Returns whether the signature for the given function should be generated."""
    if func.func.name.name.base.startswith("_"):
        return False
    if func.func.name.name.inplace:
        return False
    if func.func.name.overload_name and func.func.name.overload_name != "Tensor":
        # Ignore overloads for now.
        # Some ops only have overloaded versions, like aten::add.Tensor. And we
        # want to generate the aten::add op.
        return False
    return True


def get_op_name(func: torchgen.model.NativeFunction) -> str:
    if func.func.name.overload_name and func.func.name.overload_name != "Tensor":
        # Do not include the overload name if it is "Tensor", since ops like
        # aten::add.Tensor is what we want for aten::add.
        name = f"{func.func.name.name.base}_{func.func.name.overload_name}"
    else:
        name = f"{func.func.name.name.base}"

    # Prefix with namespace to avoid name conflicts with other operators and arguments.
    return f"{func.namespace}_{name}"


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


def create_signature(func: torchgen.model.NativeFunction) -> cg.FunctionDef:
    """Creates the signature for the given function."""
    op_name = get_op_name(func)
    args = [
        arg.argument if isinstance(arg, torchgen.model.SelfArgument) else arg
        for arg in func.func.arguments.positional
    ]
    kwargs = [
        arg
        for arg in func.func.arguments.kwarg_only
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
        return_type=create_return_type(func.func.returns),
        body=[
            cg.ThunkStmt(f'"""{func.func}"""'),
            cg.Raise(cg.Call(cg.Name("NotImplementedError"))),  # type: ignore[list-item]
        ],
    )


def create_onnx_function_module(
    functions: Sequence[torchgen.model.NativeFunction],
) -> cg.Module:
    """Creates the onnx function module."""
    return cg.Module(
        cg.ImportFrom("__future__", cg.Alias("annotations")),
        *[create_signature(func) for func in functions if should_generate_signature(func)],
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


def main(args: argparse.Namespace) -> None:
    native_functions, _ = parse_native_functions_yaml(args.yaml)
    functions: dict[str, dict[str, torchgen.model.NativeFunction]] = {}
    for func in native_functions:
        if not should_generate_signature(func):
            continue

        module_name = typing.cast(str, func.python_module)
        if not module_name:
            module_name = "core"
        if module_name not in functions:
            functions[module_name] = {}
        op_name = get_op_name(func)
        if op_name in functions[module_name]:
            logging.warning(
                "Duplicated function: %s, overload: %s", op_name, func.func.name.overload_name
            )
            continue
        functions[module_name][op_name] = func

    os.makedirs(args.outdir, exist_ok=True)

    for module_name, module_functions in functions.items():
        sorted_functions = sorted(module_functions.items(), key=lambda x: x[0])
        py_module = create_onnx_function_module([func for _, func in sorted_functions])
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
        "--yaml",
        type=str,
        help="Path to PyTorch aten/src/ATen/native/native_functions.yaml",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory for generated modules",
    )
    main(parser.parse_args())
