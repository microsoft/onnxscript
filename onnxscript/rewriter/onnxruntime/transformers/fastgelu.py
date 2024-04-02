from __future__ import annotations

import logging

import onnx

import onnxscript
from onnxscript.rewriter import function_rule

logger = logging.getLogger(__name__)


class GeluRewriteRule(function_rule.FunctionRewriteRule):
    FUNCTION_KEYWORD = "GELUActivation"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(
        self, function: onnx.FunctionProto
    ) -> tuple[onnx.FunctionProto, list[onnx.OperatorSetIdProto]]:
        del function  # Unused
        op = self.onnx_opset
        msft_opset = onnxscript.values.Opset("com.microsoft", 1)

        def gelu(input):
            return msft_opset.FastGelu(input)

        return onnxscript.script(default_opset=op)(gelu).to_function_proto(), (
            onnx.helper.make_operatorsetid("com.microsoft", 1),
        )
