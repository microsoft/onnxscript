from __future__ import annotations

import logging

import onnxscript
from onnxscript import ir
from onnxscript.rewriter import function_rule

logger = logging.getLogger(__name__)


class GeluRewriteRule(function_rule.FunctionRewriteRule):
    FUNCTION_KEYWORD = "GELUActivation"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(self, function: ir.Function) -> ir.Function:
        del function  # Unused
        op = self.onnx_opset
        msft_opset = onnxscript.values.Opset("com.microsoft", 1)

        def gelu(input):
            return msft_opset.FastGelu(input)

        function_proto = onnxscript.script(default_opset=op)(gelu).to_function_proto()
        return ir.serde.deserialize_function(function_proto)
