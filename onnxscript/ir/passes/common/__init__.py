# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "clear_metadata_and_docstring",
    "constant_manipulation",
    "inliner",
    "onnx_checker",
    "shape_inference",
    "topological_sort",
    "unused_removal",
]

from onnxscript.ir.passes.common import (
    clear_metadata_and_docstring,
    constant_manipulation,
    inliner,
    onnx_checker,
    shape_inference,
    topological_sort,
    unused_removal,
)


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
