# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for fusing Add + RMSNormalization into SkipSimplifiedLayerNormalization.

In the standard decoder layer pattern, a residual Add is followed by
RMSNormalization, and the Add output is also passed forward as the
running residual.  The ``com.microsoft::SkipSimplifiedLayerNormalization``
custom op fuses these into a single node with two outputs: the normalized
result and the skip (unnormalized sum).

These rules are **not applied by default**.  Apply them post-export::

    from mobius.rewrite_rules import skip_norm_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=skip_norm_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class AddRMSNormToSkipNorm(RewriteRuleClassBase):
    """Replace Add + RMSNormalization with SkipSimplifiedLayerNormalization.

    **Matched pattern:**

    .. code-block:: text

        norm_out = RMSNormalization(add_out, weight, epsilon=eps)

    Where ``add_out`` is produced by an ``Add`` node with at least two consumers
    (the RMSNorm and a downstream residual connection).

    **Replacement:**

    .. code-block:: text

        norm_out, _, _, skip_out = SkipSimplifiedLayerNormalization(
            add_input_0, add_input_1, weight, epsilon=eps,
        )

    ``skip_out`` (= add_input_0 + add_input_1) replaces the original ``add_out``
    in all downstream consumers except the matched RMSNorm.
    """

    def pattern(self, op, add_out, weight):
        return op.RMSNormalization(
            add_out, weight, _allow_other_attributes=True, _outputs=["norm_out"]
        )

    def check(self, context, add_out, norm_out, **_):
        result = MatchResult()

        # add_out must come from an Add node
        producer = add_out.producer()
        if producer is None or producer.op_type != "Add":
            return result.fail("Input to RMSNorm is not from an Add node")

        # The Add output must have multiple consumers (RMSNorm + residual)
        uses = list(add_out.uses())
        if len(uses) < 2:
            return result.fail(
                f"Add output has only {len(uses)} consumer(s), "
                "expected at least 2 (RMSNorm + downstream residual)"
            )

        # Verify RMSNormalization has epsilon attribute
        rmsnorm = norm_out.producer()
        if rmsnorm.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute on RMSNormalization")

        return result

    def rewrite(self, op, add_out, weight, norm_out, **_):
        rmsnorm = norm_out.producer()
        epsilon = rmsnorm.attributes.get_float("epsilon")

        # Get the two inputs of the Add node
        add_node = add_out.producer()
        input_a = add_node.inputs[0]
        input_b = add_node.inputs[1]

        outputs = op.op_multi_out(
            "SkipSimplifiedLayerNormalization",
            inputs=[input_a, input_b, weight],
            domain="com.microsoft",
            attributes={"epsilon": epsilon},
            num_outputs=4,
        )
        new_norm_out = outputs[0]
        skip_out = outputs[3]

        # Replace add_out with skip_out in all other consumers
        add_out.replace_all_uses_with(skip_out)

        return new_norm_out


def skip_norm_rules() -> RewriteRuleSet:
    """Return rules that fuse Add + RMSNorm into SkipSimplifiedLayerNormalization.

    These rules match the residual Add + RMSNormalization pattern common in
    decoder layers and replace it with the fused Microsoft
    ``SkipSimplifiedLayerNormalization`` custom op.

    Returns:
        :class:`RewriteRuleSet` containing the Add+RMSNorm fusion rule.
    """
    return RewriteRuleSet([AddRMSNormToSkipNorm().rule()])
