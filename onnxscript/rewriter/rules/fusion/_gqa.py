# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Union

import onnx_ir as ir

import onnxscript.rewriter._fusion_utils as _fusion_utils
from onnxscript.rewriter import _basics, pattern

Dim = Union[int, ir.SymbolicDim]


class OnnxGroupQueryAttention(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__("ONNXGQA", remove_nodes=False)

    def pattern(
        self,
        op,
        query_BHSD,
        key_BHkvSD,
        value_BHkvSD,
        past_key_BHkvSpD,
        past_value_BHkvSpD,
    ):
        # Concatenate past_key cache and current key, expand across heads
        # that share key/value.

        present_key_BHkvStD = op.Concat(past_key_BHkvSpD, key_BHkvSD, axis=-2)
        present_key_BHkv1StD = op.Unsqueeze(present_key_BHkvStD, 2)
        present_key_BHkvGStD = op.Expand(present_key_BHkv1StD, pattern.ANY_VALUE)
        present_key_BHStD = op.Reshape(
            present_key_BHkvGStD, pattern.ANY_VALUE, _outputs=["present_key_BHStD"]
        )

        # Concatenate past_value cache and current value, expand across heads
        # that share key/value.
        present_value_BHkvStD = op.Concat(past_value_BHkvSpD, value_BHkvSD, axis=-2)
        present_value_BHkv1StD = op.Unsqueeze(present_value_BHkvStD, 2)
        present_value_BHkvGStD = op.Expand(present_value_BHkv1StD, pattern.ANY_VALUE)
        present_value_BHStD = op.Reshape(
            present_value_BHkvGStD, pattern.ANY_VALUE, _outputs=["present_value_BHStD"]
        )

        attention_BHSDh = op.Attention(
            query_BHSD,
            present_key_BHStD,
            present_value_BHStD,
            pattern.Var("mask", can_match_none=True),
            _outputs=["attention_BHSDh"],
        )

        return attention_BHSDh, present_key_BHkvStD, present_value_BHkvStD

    def check(
        self,
        context: _basics.MatchContext,
        query_BHSD,
        key_BHkvSD,
        value_BHkvSD,
        past_key_BHkvSpD,
        past_value_BHkvSpD,
        present_key_BHStD,
        present_value_BHStD,
        **_,
    ):
        bindings: dict[str, Dim] = {}
        # Check that inputs to new Attention node have expected shapes
        _fusion_utils.check_shape(bindings, query_BHSD, ["B", "H", "S", "D"])
        _fusion_utils.check_shape(bindings, key_BHkvSD, ["B", "Hkv", "S", "D"])
        _fusion_utils.check_shape(bindings, value_BHkvSD, ["B", "Hkv", "S", "D"])
        _fusion_utils.check_shape(bindings, past_key_BHkvSpD, ["B", "Hkv", "P", "D"])
        _fusion_utils.check_shape(bindings, past_value_BHkvSpD, ["B", "Hkv", "P", "D"])
        # We need to check that the Expand/Reshape arguments are as expected.
        # As a substitute, we check that the outputs of Expand=>Reshape have expected shapes.
        # TODO (rama): May be better to check the actual Expand/Reshape arguments.
        _fusion_utils.check_shape(bindings, present_key_BHStD, ["B", "H", "S+P", "D"])
        _fusion_utils.check_shape(bindings, present_value_BHStD, ["B", "H", "S+P", "D"])

        return True

    def rewrite(
        self,
        op,
        query_BHSD,
        key_BHkvSD,
        value_BHkvSD,
        past_key_BHkvSpD,
        past_value_BHkvSpD,
        mask,
        attention_BHSDh,
        **_,
    ):
        original_attention_node = attention_BHSDh.producer()
        original_attrs = original_attention_node.attributes
        return op.Attention(
            query_BHSD,
            key_BHkvSD,
            value_BHkvSD,
            mask,
            past_key_BHkvSpD,
            past_value_BHkvSpD,
            **original_attrs,
            _outputs=3,
        )


_basic_gqa_rule = OnnxGroupQueryAttention.rule()

gqa_rules = pattern.RewriteRuleSet([_basic_gqa_rule])

fuse_gqa = _fusion_utils.apply_fusion_rules(gqa_rules)
