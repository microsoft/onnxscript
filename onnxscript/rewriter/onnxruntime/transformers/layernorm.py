from __future__ import annotations

import logging

import onnx
from onnx import numpy_helper

import onnxscript
from onnxscript.rewriter import function_rule

logger = logging.getLogger(__name__)


class LNRewriteRule(function_rule.FunctionRewriteRule):
    FUNCTION_KEYWORD = "layernorm"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(  # type: ignore[misc]
        self, function: onnx.FunctionProto
    ) -> tuple[onnx.FunctionProto, list[onnx.OperatorSetIdProto]]:
        # TODO(bowbao): Might be more desirable to annotate as attribute in nn.Module
        aten_add_node = self._find_node_by_type(function, "", "Add")
        if aten_add_node is None:
            raise function_rule.FunctionRewriteError("Could not find Add node")

        eps_node = self._find_constant_node(function, aten_add_node.input[1])
        if eps_node is None:
            raise function_rule.FunctionRewriteError("Could not find eps node")

        eps = numpy_helper.to_array(eps_node.attribute[0].t).item()
        logger.info("eps: %s", eps)

        # TODO(ORT): SimplifiedLayerNormalization in ort is defined under onnx domain.
        # https://github.com/microsoft/onnxruntime/issues/7573
        # msft_op = onnxscript.values.Opset("com.microsoft", 1)
        op = self.onnx_opset

        def ln(input, weight):
            return op.SimplifiedLayerNormalization(
                input, weight, axis=-1, epsilon=eps, stash_type=1
            )

        return onnxscript.script(default_opset=op)(ln).to_function_proto(), []
