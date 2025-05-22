# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__all__ = [
    "PassBase",
    "PassResult",
    "PassManager",
    "Sequential",
    "InPlacePass",
    "FunctionalPass",
    # Errors
    "InvariantError",
    "PreconditionError",
    "PostconditionError",
    "PassError",
]

from onnx_ir.passes import (
    FunctionalPass,
    InPlacePass,
    InvariantError,
    PassBase,
    PassError,
    PassManager,
    PassResult,
    PostconditionError,
    PreconditionError,
    Sequential,
)
