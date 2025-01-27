# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

import onnxscript.optimizer._constant_folding as constant_folding
import onnxscript.optimizer._legacy._optimizer as legacy_optimizer
import onnxscript.optimizer._legacy.constant_folding as legacy_constant_folding
from onnxscript import ir
from onnxscript.optimizer._optimizer import optimize_ir
from onnxscript.optimizer._remove_unused import remove_unused_nodes

basic_constant_propagation = constant_folding.basic_constant_propagation
fold_constants_ir = constant_folding.fold_constants


def optimize(model: ir.Model | onnx.ModelProto, *args, **kwargs):
    if isinstance(model, ir.Model):
        return optimize_ir(model, *args, **kwargs)
    else:
        return legacy_optimizer.optimize(model, *args, **kwargs)


def fold_constants(model: ir.Model | onnx.ModelProto, *args, **kwargs):
    if isinstance(model, ir.Model):
        return constant_folding.fold_constants(model, *args, **kwargs)
    else:
        return legacy_constant_folding.fold_constants(model, *args, **kwargs)


__all__ = [
    "fold_constants",
    "fold_constants_ir",
    "remove_unused_nodes",
    "optimize",
    "optimize_ir",
    "basic_constant_propagation",
]
