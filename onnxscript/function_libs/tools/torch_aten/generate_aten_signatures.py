"""Generates the ATen signatures for the ONNX ATen operator set using torch.ops."""
from __future__ import annotations

import argparse
import ast
import sys
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
    if arg_type in {
        "float",
        "int",
        "bool",
    }:
        # Convert python type to onnx tensor types
        arg_type = arg_type.upper()
    if arg.type.size is None:
        return cg.TypeRef(None, arg_type, cg.EllipsisTypeRef())
    return cg.TypeRef(None, arg_type, *[cg.TypeRef(None, f"{arg.type.size}")])


def arg_type_to_str(arg_type: torchgen.model.Type) -> str:
    if arg_type.is_base_ty_like(torchgen.model.BaseTy.Tensor):
        return "TensorType"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.SymInt):
        return "INT64"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.Scalar):
        return "TensorType"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.float):
        return "float"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.int):
        return "int"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.bool):
        return "bool"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.str):
        return "str"
    elif arg_type.is_base_ty_like(torchgen.model.BaseTy.ScalarType):
        return "int"
    else:
        return "Any"


def get_argument_type(arg: torchgen.model.Argument) -> cg.TypeRef:
    """Returns the Python type for the given argument."""
    if isinstance(arg.type, torchgen.model.ListType):
        inner_type = create_list_type(arg)
    else:
        inner_type = cg.TypeRef(None, arg_type_to_str(arg.type))

    if arg.type.is_nullable():
        return cg.TypeRef(None, "Optional", inner_type)
    return inner_type


def should_generate_signature(func: torchgen.model.NativeFunction) -> bool:
    """Returns whether the signature for the given function should be generated."""
    if func.func.name.name.base.startswith("_"):
        return False
    if func.func.name.overload_name:
        # Ignore overloads for now.
        return False
    return True


def get_op_name(func: torchgen.model.NativeFunction) -> str:
    if func.func.name.overload_name:
        return f"{func.func.name.name.base}_{func.func.name.overload_name}"

    return func.func.name.name.base  # type: ignore[no-any-return]


def format_default_value(default: str) -> str:
    if default.startswith("[") and default.endswith("]"):
        # Convert list to tuple
        default_val = ast.literal_eval(default)
        assert isinstance(default_val, list)
        return repr(tuple(default_val))
    return default


def create_return_type(returns: Sequence[torchgen.model.Return]) -> cg.TypeRef:
    """Returns the Python type for the return value of the given function."""
    if not returns:
        return cg.BuiltinTypeRef("Any")
    return_nodes = []
    for return_val in returns:
        return_type = return_val.type
        return_node = cg.TypeRef(None, arg_type_to_str(return_type))
        if return_type.is_nullable():
            return_node = cg.TypeRef(None, "Optional", return_node)
        return_nodes.append(return_node)
    if len(return_nodes) == 1:
        return return_nodes[0]
    return cg.BuiltinTypeRef("tuple", *return_nodes)


def create_signature(func: torchgen.model.NativeFunction) -> Any:
    """Creates the signature for the given function."""
    print(func.namespace)
    print(func.python_module)
    print(func.func.name.overload_name)

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
            arg.name,
            get_argument_type(arg),
            default_value=cg.ThunkExpr(format_default_value(arg.default))
            if arg.default is not None
            else None,
        )
        for arg in args
    ]
    if kwargs:
        py_args += [
            # Arguments after this point are keyword-only.
            cg.Arg(
                "*",
            )
        ] + [
            cg.Arg(
                kwarg.name,
                get_argument_type(kwarg),
                default_value=cg.ThunkExpr(format_default_value(kwarg.default or "None")),
                is_kwarg=True,
            )
            for kwarg in kwargs
        ]

    return cg.FunctionDef(
        op_name,
        *py_args,
        return_type=create_return_type(func.func.returns),
        body=[cg.Raise(cg.Call(cg.Name("NotImplementedError")))],  # type: ignore[list-item]
    )


def main(args: argparse.Namespace) -> None:
    native_functions, _ = parse_native_functions_yaml(args.native_functions_yaml)

    for func in native_functions:
        if not should_generate_signature(func):
            continue

        py_tree = create_signature(func)
        py_tree.accept(cg.PythonWriter(sys.stdout))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-functions-yaml",
        type=str,
        default="/home/justinchu/dev/pytorch/aten/src/ATen/native/native_functions.yaml",
    )
    main(parser.parse_args())
