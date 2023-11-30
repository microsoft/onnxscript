from __future__ import annotations

from typing import TypeVar

import onnx

import onnxscript
from onnxscript.onnx_types import (
    DOUBLE,
    FLOAT,
    FLOAT16,
)
from onnxscript.values import Op, Opset

_DOMAIN = "com.microsoft"
_VERSION = 1


class MicrosoftOpset(Opset):
    def __new__(cls):
        return Opset.__new__(cls, _DOMAIN, _VERSION)

    T_Inverse = TypeVar("T_Inverse", FLOAT16, FLOAT, DOUBLE)

    inverse_schema = onnx.defs.OpSchema(
        "Inverse",
        _DOMAIN,
        _VERSION,
        inputs=(
            onnx.defs.OpSchema.FormalParameter(
                "X",
                "T",
                description="Input tensor. Every matrix in the batch must be invertible.",
            ),
        ),
        outputs=(
            onnx.defs.OpSchema.FormalParameter(
                "Y",
                "T",
                description="Output tensor of the same type and shape as the input tensor.",
            ),
        ),
        type_constraints=(
            onnxscript.values.TypeConstraint("T", ["float16", "float", "double"]).as_tuple(),
        ),
    )

    def Inverse(self, X: T_Inverse) -> T_Inverse:
        """Takes the inverse of the square matrix(s) in X.

        Args:
            X: Input tensor. Every matrix in the batch must be invertible.

        Returns:
            Output tensor of the same type and shape as the input tensor.
        """

        op = Op(self, "Inverse", self.inverse_schema)
        return op(*self._prepare_inputs(self.inverse_schema, X))


microsoft_opset = MicrosoftOpset()
