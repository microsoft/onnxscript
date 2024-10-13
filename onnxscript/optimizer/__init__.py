# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

import onnxscript.optimizer._legacy._optimizer as legacy_optimizer
from onnxscript import ir
from onnxscript.optimizer._constant_folding import basic_constant_propagation
from onnxscript.optimizer._legacy.constant_folding import fold_constants
from onnxscript.optimizer._optimizer import optimize_ir
from onnxscript.optimizer._remove_unused import remove_unused_nodes


def optimize(model: ir.Model | onnx.ModelProto, *args, **kwargs):
    if isinstance(model, ir.Model):
        return optimize_ir(model, *args, **kwargs)
    else:
        return legacy_optimizer.optimize(model, *args, **kwargs)


__all__ = [
    "fold_constants",
    "remove_unused_nodes",
    "optimize",
    "optimize_ir",
    "basic_constant_propagation",
]
