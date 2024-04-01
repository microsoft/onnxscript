from __future__ import annotations

from onnxrewriter.rewriter import function_rule
from onnxrewriter.rewriter.onnxruntime.transformers import (
    fastgelu,
    layernorm,
    multihead_attention,
)

TRANSFORMERS_FUNCTION_REWRITE_RULES: list[type[function_rule.FunctionRewriteRule]] = [
    multihead_attention.GQALlama2RewriteRule,
    multihead_attention.GQALlamaSdpa2RewriteRule,
    multihead_attention.AttnPhi15RewriteRule,
    layernorm.LNRewriteRule,
    fastgelu.GeluRewriteRule,
]
