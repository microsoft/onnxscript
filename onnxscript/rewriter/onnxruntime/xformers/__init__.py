# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    "fuse_rms_normalization",
    "fuse_normalization",
    "fuse_rotary_embedding",
    "fuse_cos_sin_cache",
    "fuse_sdpa",
    "fuse_mha",
    "fuse_xformers",
]

from onnxscript.rewriter.onnxruntime.xformers.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.onnxruntime.xformers.fuse_xformers import fuse_xformers
from onnxscript.rewriter.onnxruntime.xformers.mha import fuse_mha
from onnxscript.rewriter.onnxruntime.xformers.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.onnxruntime.xformers.rotary_embedding import fuse_rotary_embedding
from onnxscript.rewriter.onnxruntime.xformers.sdpa import fuse_sdpa
from onnxscript.rewriter.onnxruntime.xformers.skip_normalization import fuse_normalization
