# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "PassBase",
    "PassResult",
    "PassManager",
    # Errors
    "InvariantError",
    "PreconditionError",
    "PostconditionError",
    "PassError",
]

from onnxscript.ir.passes._pass_infra import (
    InvariantError,
    PassBase,
    PassError,
    PassManager,
    PassResult,
    PostconditionError,
    PreconditionError,
)


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
