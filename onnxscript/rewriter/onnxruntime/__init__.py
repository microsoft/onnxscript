from __future__ import annotations

import onnx

from onnxscript import ir
from onnxscript.optimizer import remove_unused, remove_unused_function
from onnxscript.rewriter import function_rule, pattern
from onnxscript.rewriter.onnxruntime import (
    group_normalization_merge_silu,
    instance_to_group_normalization,
    softmax,
    transformers,
)
from onnxscript.rewriter.onnxruntime.bfloat16_utils import bfloat16_converter

ORT_FUNCTION_REWRITE_RULES = [*transformers.TRANSFORMERS_FUNCTION_REWRITE_RULES]

ORT_PATTERN_REWRITE_RULES = [
    *softmax.rules.rules,
    *instance_to_group_normalization.rules.rules,
    # NOTE: group normalization merge silu should be applied after instance to group normalization
    *group_normalization_merge_silu.rules.rules,
]


def rewrite(
    model_proto: onnx.ModelProto,
    /,
    function_rules: list[type[function_rule.FunctionRewriteRule]] | None = None,
    pattern_rules: list[pattern.RewriteRule] | None = None,
    convert_bloat16_to_float16: bool = False,
) -> onnx.ModelProto:
    """Rewrite the model using the given rules.

    Args:
        model_proto: The model to rewrite.
        function_rules: The function rewrite rules to apply. If None, the default rules
            for onnxruntime are used.
        pattern_rules: The pattern rewrite rules to apply. If None, the default rules
            for onnxruntime are used.
        convert_bloat16_to_float16: If True, convert bfloat16 to float16.

    Returns:
        The rewritten model.
    """
    function_rules = function_rules or ORT_FUNCTION_REWRITE_RULES
    pattern_rules = pattern_rules or ORT_PATTERN_REWRITE_RULES
    model = ir.serde.deserialize_model(model_proto)
    # TODO(bowenbao): Function rules first, or pattern rules first?
    if function_rules:
        for rule_cls in function_rules:
            count, model = rule_cls().apply_to_model(model)
            print(f"Applied {count} of onnxruntime specific function rewrite rules.")
    if pattern_rules:
        count = pattern.RewriteRuleSet(pattern_rules).apply_to_model(model)
        print(f"Applied {count} of onnxruntime specific pattern rewrite rules.")

    # TODO: remove this after onnxruntime supports bfloat16
    if convert_bloat16_to_float16:
        bfloat16_converter.dtype_adapter_for_bfloat16_model(model)

    model_proto = ir.serde.serialize_model(model)
    remove_unused.remove_unused_nodes(model_proto)
    remove_unused_function.remove_unused_functions(model_proto)
    return model_proto
