# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Rules to collapse Transpose nodes into initializers."""

from __future__ import annotations

import logging

import numpy as np

from onnxscript import ir
from onnxscript.rewriter import _ir_utils as ir_utils
from onnxscript.rewriter import pattern as orp

logger = logging.getLogger(__name__)


class TransposeInitializer(orp.RewriteRuleClassBase):
    """Folds Transpose nodes into initializers."""

    def __init__(self):
        super().__init__("TransposeInitializer", remove_nodes=True)

    def pattern(self, op, initializer):
        return op.Transpose(initializer, _allow_other_attributes=True)

    def rewrite(self, op, initializer: ir.Value) -> ir.Value:
        original_transpose = initializer.consumers()[0]
        perm_attr = original_transpose.attributes.get("perm")
        assert isinstance(perm_attr, ir.Attr)
        array = ir_utils.get_numpy_value(initializer)
        if array is None:
            # Do nothing
            logger.debug("Failed to obtain the initializer value. Do nothing")
            # perm=None is filtered out when the attribute is constructed so we are ok
            return op.Transpose(initializer, perm=perm_attr)

        if perm_attr is not None:
            perm = perm_attr.as_ints()
        else:
            perm = None
        transposed = np.transpose(array, axes=perm)
        new_name = f"{initializer.const_value.name}_transposed"
        return op.initializer(ir.tensor(transposed, name=new_name))

    def check(self, context, initializer: ir.Value) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()
        if initializer.const_value is None:
            return check_result.fail("Value is not an initializer, const_value is None")
        if initializer.producer() is not None:
            return check_result.fail("Value is not an initializer, producer is not None")
        if initializer.uses() != 1:
            return check_result.fail("Initializer is used by more than one node")
        return check_result


rule = TransposeInitializer.rule()
