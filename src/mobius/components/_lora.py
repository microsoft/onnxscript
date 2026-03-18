# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA (Low-Rank Adaptation) components."""

from __future__ import annotations

import numpy
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Linear


class LoRALinear(Linear):
    """Linear layer with LoRA (Low-Rank Adaptation) adapters.

    Each adapter adds a low-rank contribution: ``(x @ A^T) @ B^T * scale``.
    Multiple named adapters (e.g. "vision", "speech") are summed.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        lora_adapters: List of ``(name, rank, scale)`` tuples.
            Each adapter creates ``lora_A.{name}.weight`` and
            ``lora_B.{name}.weight`` parameters matching HuggingFace naming.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lora_adapters: list[tuple[str, int, float]] | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self._adapters: list[tuple[str, nn.Parameter]] = []
        if lora_adapters:
            for name, rank, scale in lora_adapters:
                # Parameters are named to match HuggingFace convention:
                #   lora_A.{name}.weight  and  lora_B.{name}.weight
                setattr(
                    self,
                    f"_lora_A_{name}",
                    nn.Parameter([rank, in_features], name=f"lora_A.{name}.weight"),
                )
                setattr(
                    self,
                    f"_lora_B_{name}",
                    nn.Parameter([out_features, rank], name=f"lora_B.{name}.weight"),
                )
                # Store scale as a named Parameter so each module gets a unique
                # initializer (avoids constant name collisions in the graph).
                scale_param = nn.Parameter(
                    [],
                    name=f"lora_scale.{name}",
                    data=ir.tensor(numpy.array(scale, dtype=numpy.float32)),
                )
                setattr(self, f"_lora_scale_{name}", scale_param)
                self._adapters.append((name, scale_param))

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        result = super().forward(op, x)
        for name, scale_param in self._adapters:
            lora_a = getattr(self, f"_lora_A_{name}")
            lora_b = getattr(self, f"_lora_B_{name}")
            h = op.MatMul(x, op.Transpose(lora_a, perm=[1, 0]))
            lora_out = op.MatMul(h, op.Transpose(lora_b, perm=[1, 0]))
            lora_out = op.Mul(lora_out, scale_param)
            result = op.Add(result, lora_out)
        return result
