"""Generates the ATen signatures for the ONNX ATen operator set using torch.ops."""
from __future__ import annotations

import argparse
import sys
from typing import Any

import torchgen.gen
import torchgen.model
import yaml

import opgen.pygen as cg

# BaseTy = Enum(
#     "BaseTy",
#     (
#         "Generator",
#         "ScalarType",
#         "Tensor",
#         "int",
#         "Dimname",
#         "DimVector",
#         "float",
#         "str",
#         "bool",
#         "Layout",
#         "Device",
#         "Scalar",
#         "MemoryFormat",
#         "QScheme",
#         "Storage",
#         "Stream",
#         "SymInt",
#         "ConstQuantizerPtr",
#     ),
# )


def load_native_function_yaml(yaml_path: str):
    with open(yaml_path, encoding="utf-8") as f:
        yaml_str = f.read()
    with open(yaml_path, encoding="utf-8") as f:
        all_functions = yaml.safe_load(f)
    valid_tags = set()
    for func in all_functions:
        if "tags" not in func:
            continue
        valid_tags.add(func["tags"])

    return yaml_str, valid_tags


def parse_native_functions_yaml(yaml_path: str) -> tuple[Any, Any]:
    yaml_str, valid_tags = load_native_function_yaml(yaml_path)
    yaml_struct = yaml.load(yaml_str, Loader=torchgen.gen.LineLoader)
    parsed = torchgen.gen.parse_native_yaml_struct(
        yaml_struct, valid_tags, path=yaml_path, skip_native_fns_gen=True
    )
    return parsed.native_functions, parsed.backend_indices


def get_argument_type(arg: torchgen.model.Argument) -> cg.TypeRef:
    # TODO: Handel scalar type
    optional = arg.type.is_nullable()
    if arg.type.is_base_ty_like(torchgen.model.BaseTy.Tensor):
        inner_type = cg.TypeRef(None, "Tensor")
    elif arg.type.is_base_ty_like(torchgen.model.BaseTy.SymInt):
        # TODO(justinchuby): Make sure this is a scalar
        inner_type = cg.TypeRef(None, "INT64")
    elif arg.type.is_base_ty_like(torchgen.model.BaseTy.float):
        inner_type = cg.TypeRef(None, "float")
    elif arg.type.is_base_ty_like(torchgen.model.BaseTy.int):
        inner_type = cg.TypeRef(None, "int")
    elif arg.type.is_base_ty_like(torchgen.model.BaseTy.bool):
        inner_type = cg.TypeRef(None, "bool")
    else:
        inner_type = cg.TypeRef(None, "Any")

    if optional:
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

    return func.func.name.name.base


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
            default_value=cg.ThunkExpr(arg.default) if arg.default is not None else None,
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
                default_value=cg.ThunkExpr(kwarg.default or "None"),
                is_kwarg=True,
            )
            for kwarg in kwargs
        ]

    return cg.FunctionDef(
        op_name,
        *py_args,
        return_type=None,  # TODO: Add return type
        body=[
            cg.Raise(
                cg.ThunkExpr("NotImplementedError"),
            )
        ],
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
