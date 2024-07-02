# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

import onnxscript.optimizer.remove_unused_ir
import onnxscript.optimizer.remove_unused_proto
from onnxscript import ir


def remove_unused_nodes(model: ir.Model | onnx.ModelProto) -> None:
    if isinstance(model, ir.Model):
        onnxscript.optimizer.remove_unused_ir.remove_unused_nodes(model)
    else:
        onnxscript.optimizer.remove_unused_proto.remove_unused_nodes(model)
