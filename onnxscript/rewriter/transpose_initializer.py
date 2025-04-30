"""Rules to collapse Transpose nodes into initializers."""
from __future__ import annotations
from onnxscript import ir
from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter import pattern as orp

import logging

logger = logging.getLogger(__name__)

class TransposeInitializer(orp.RewriteRuleClassBase):
    """Folds Transpose nodes into initializers."""

    def __init__(self):
        super().__init__("TransposeInitializer", remove_nodes=True)

    def pattern(self, op, initializer):
        return op.Transpose(initializer, _allow_other_attributes=True)

    def rewrite(self, op, initializer: ir.Value) -> ir.Value:
        array = ir_utils.get_const_value(initializer)
        if array is None:
            # Do nothing
            logger.debug("Failed to obtain the initializer value. Do nothing")
            # TODO: Handle both when perms is None and when perms is not None
            return op.Transpose(initializer, perms)
        # TODO Obtain perms from the matched node
        return op.initializer(ir.tensor())

    def check(self, context, initializer: ir.Value) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()
        if initializer.const_value is None:
            return check_result.fail("Value is not an initializer, const_value is None")
        if initializer.producer() is not None:
            return check_result.fail("Value is not an initializer, producer is not None")
        return check_result


rule = TransposeInitializer.rule()
