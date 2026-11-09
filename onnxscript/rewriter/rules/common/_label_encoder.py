# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Compose exact string/integer label mappings."""

from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _mapping(node: ir.Node):
    attrs = node.attributes
    if any(attr.is_ref() for attr in attrs.values()):
        return None
    key_names = [name for name in attrs if name.startswith("keys_")]
    value_names = [name for name in attrs if name.startswith("values_")]
    if len(key_names) != 1 or len(value_names) != 1:
        return None
    key_name, value_name = key_names[0], value_names[0]
    supported = {"keys_strings", "keys_int64s", "values_strings", "values_int64s"}
    if key_name not in supported or value_name not in supported:
        return None
    keys = (
        list(attrs[key_name].as_strings())
        if key_name == "keys_strings"
        else list(attrs[key_name].as_ints())
    )
    values = (
        list(attrs[value_name].as_strings())
        if value_name == "values_strings"
        else list(attrs[value_name].as_ints())
    )
    if len(keys) != len(values) or len(set(keys)) != len(keys):
        return None
    default_name = "default_string" if value_name == "values_strings" else "default_int64"
    fallback = attrs.get(default_name)
    if value_name == "values_strings":
        default = fallback.as_string() if fallback is not None else "_Unused"
    else:
        default = fallback.as_int() if fallback is not None else -1
    return key_name, keys, value_name, values, default_name, default


class ComposeLabelEncoders(RewriteRuleClassBase):
    def pattern(self, op, x):
        return op.LabelEncoder(op.LabelEncoder(x, _domain="ai.onnx.ml"), _domain="ai.onnx.ml")

    def check(self, context, x) -> MatchResult:
        result = MatchResult()
        version = context.graph_or_function.opset_imports.get(
            "ai.onnx.ml", context.model.opset_imports.get("ai.onnx.ml", 0)
        )
        if version < 2:
            return result.fail("List-based LabelEncoder mappings require ai.onnx.ml opset 2.")
        first = context.root.inputs[0].producer()
        inner, outer = _mapping(first), _mapping(context.root)
        if inner is None or outer is None:
            return result.fail(
                "Mappings must have distinct string/int64 keys and list values."
            )
        key_name, keys, value_name, values, _, default = inner
        outer_key, outer_keys, outer_value, outer_values, default_name, outer_default = outer
        if value_name.removeprefix("values_") != outer_key.removeprefix("keys_"):
            return result.fail("The intermediate mapping types do not match.")
        lookup = dict(zip(outer_keys, outer_values))
        mapped = [lookup.get(value, outer_default) for value in values]
        self._attributes = {
            key_name: (
                ir.AttrStrings(key_name, keys)
                if key_name == "keys_strings"
                else ir.AttrInt64s(key_name, keys)
            ),
            outer_value: (
                ir.AttrStrings(outer_value, mapped)
                if outer_value == "values_strings"
                else ir.AttrInt64s(outer_value, mapped)
            ),
            default_name: lookup.get(default, outer_default),
        }
        return result

    def rewrite(self, op, x):
        return op.LabelEncoder(x, _domain="ai.onnx.ml", **self._attributes)


compose_label_encoders_rule = ComposeLabelEncoders.rule(remove_nodes=False)
rules = RewriteRuleSet([compose_label_encoders_rule])
