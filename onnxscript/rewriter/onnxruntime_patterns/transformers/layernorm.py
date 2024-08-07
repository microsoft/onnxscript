# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnxscript
from onnxscript import ir
from onnxscript.rewriter import _ir_utils, function_rule

logger = logging.getLogger(__name__)


class LNRewriteRule(function_rule.FunctionRewriteRule):
    FUNCTION_KEYWORD = "layernorm"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(self, function: ir.Function) -> ir.Function:
        # TODO(bowbao): Might be more desirable to annotate as attribute in nn.Module
        aten_add_node = self._find_node_by_type(function, "", "Add")
        if aten_add_node is None:
            raise function_rule.FunctionRewriteError("Could not find Add node")

        eps_ir_value = _ir_utils.propagate_const_value(aten_add_node.inputs[1])
        eps_const_value = eps_ir_value.const_value
        if eps_const_value is None:
            raise function_rule.FunctionRewriteError("Could not find eps")
        eps_numpy_value = eps_const_value.numpy()
        eps = eps_numpy_value.item()
        logger.info("eps: %s", eps)

        # TODO(ORT): SimplifiedLayerNormalization in ort is defined under onnx domain.
        # https://github.com/microsoft/onnxruntime/issues/7573
        # msft_op = onnxscript.values.Opset("com.microsoft", 1)
        op = self.onnx_opset

        def ln(input, weight):
            return op.SimplifiedLayerNormalization(
                input, weight, axis=-1, epsilon=eps, stash_type=1
            )

        function_proto = onnxscript.script(default_opset=op)(ln).to_function_proto()
        return ir.serde.deserialize_function(function_proto)
