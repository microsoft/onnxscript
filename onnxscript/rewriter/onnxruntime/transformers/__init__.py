from __future__ import annotations

from onnxscript.rewriter import function_rule
from onnxscript.rewriter.onnxruntime.transformers import (
    biassplitgelu,
    fastgelu,
    layernorm,
    multihead_attention,
)

TRANSFORMERS_FUNCTION_REWRITE_RULES: list[type[function_rule.FunctionRewriteRule]] = [
    multihead_attention.GQALlama2RewriteRule,
    multihead_attention.GQALlamaSdpa2RewriteRule,
    multihead_attention.AttnPhi15RewriteRule,
    multihead_attention.MHAStableDiffusionUnetRewriteRule,
    layernorm.LNRewriteRule,
    fastgelu.GeluRewriteRule,
    biassplitgelu.GegluRewriteRule,
]
