# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnxscript
from onnxscript import ir
from onnxscript.rewriter import function_rule

logger = logging.getLogger(__name__)


class GegluRewriteRule(function_rule.FunctionRewriteRule):
    FUNCTION_KEYWORD = "GEGLU"
    PACKAGE_NAME = "diffusers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()  # type: ignore[misc]
    def _fusion(self, function: ir.Function) -> ir.Function:
        del function  # Unused
        op = self.onnx_opset
        msft_opset = onnxscript.values.Opset("com.microsoft", 1)

        def ggelu(input, weight, bias):
            weight_transpose = op.Transpose(weight, [1, 0])
            matmul_input = op.MatMul(input, weight_transpose)
            return msft_opset.BiasSplitGelu(matmul_input, bias)

        function_proto = onnxscript.script(default_opset=op)(ggelu).to_function_proto()  # type: ignore[arg-type]
        return ir.serde.deserialize_function(function_proto)
