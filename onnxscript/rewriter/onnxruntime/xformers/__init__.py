# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter.onnxruntime.xformers.multi_head_attention import (
    mha_rules as mha_rules,
)
from onnxscript.rewriter.onnxruntime.xformers.rms_normalization import (
    rms_normalization_rules as rms_normalization_rules,
)
from onnxscript.rewriter.onnxruntime.xformers.rotary_embedding import (
    rotary_embedding_rules as rotary_embedding_rules,
)
from onnxscript.rewriter.onnxruntime.xformers.sdpa import sdpa_rules as sdpa_rules
from onnxscript.rewriter.onnxruntime.xformers.skip_normalization import (
    skip_normalization_rules as skip_normalization_rules,
)
