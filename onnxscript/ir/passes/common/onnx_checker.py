# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Passes for debugging purposes."""

from __future__ import annotations

__all__ = [
    "CheckerPass",
]

import onnx

from onnxscript import ir
from onnxscript.ir.passes.common import _c_api_utils


class CheckerPass(ir.passes.PassBase):
    """Run onnx checker on the model."""

    @property
    def in_place(self) -> bool:
        return True

    @property
    def changes_input(self) -> bool:
        return False

    def __init__(
        self,
        full_check: bool = False,
        skip_opset_compatibility_check: bool = False,
        check_custom_domain: bool = False,
    ):
        super().__init__()
        self.full_check = full_check
        self.skip_opset_compatibility_check = skip_opset_compatibility_check
        self.check_custom_domain = check_custom_domain

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """Run the onnx checker on the model."""

        def _partial_check_model(proto: onnx.ModelProto) -> None:
            """Partial function to check the model."""
            onnx.checker.check_model(
                proto,
                full_check=self.full_check,
                skip_opset_compatibility_check=self.skip_opset_compatibility_check,
                check_custom_domain=self.check_custom_domain,
            )

        _c_api_utils.call_onnx_api(func=_partial_check_model, model=model)
        # The model is not modified
        return ir.passes.PassResult(model, False)
