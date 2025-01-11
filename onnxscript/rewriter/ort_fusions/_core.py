# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.optimizer import remove_unused_nodes, optimize
from onnxscript.rewriter.onnxruntime.xformers.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.onnxruntime.xformers.mha import fuse_mha
from onnxscript.rewriter.onnxruntime.xformers.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.onnxruntime.xformers.rotary_embedding import fuse_rotary_embedding
from onnxscript.rewriter.onnxruntime.xformers.sdpa import fuse_sdpa
from onnxscript.rewriter.onnxruntime.xformers.skip_normalization import fuse_normalization


def fuse_xformers(model: ir.Model) -> None:
    optimize(model)
    fuse_rms_normalization(model)
    fuse_normalization(model)
    fuse_rotary_embedding(model)
    fuse_cos_sin_cache(model)
    fuse_sdpa(model)
    fuse_mha(model)
    remove_unused_nodes(model)

def optimize_for_ort(model: ir.Model) -> None:
    # TODO(rama): Include the other optimizations
    fuse_xformers(model)