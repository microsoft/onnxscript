# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for ConvAffineFusion and AffineConvFusion rules.

Adds coverage for: non-constant weight (negative), non-scalar scale (negative),
padded pre-conv affine (negative), non-constant bias (negative).
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import rewrite
from onnxscript.rewriter.rules.common import (
    affine_conv_fusion_rule,
    conv_affine_fusion_rule,
)


class FuseConvAffineExtendedTest(unittest.TestCase):
    """Extended tests for ConvAffineFusion and AffineConvFusion."""

    def _build_conv_affine_model(
        self,
        *,
        w_is_const: bool = True,
        b_is_const: bool = True,
        scale_is_scalar: bool = True,
        pads: tuple = (1, 1, 1, 1),
    ) -> ir.Model:
        """Build Conv → Mul(scale) → Add(offset) model with configurable constants."""
        tape = ir.tape.Tape()
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32]))

        if w_is_const:
            w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        else:
            w = ir.val("w", dtype=ir.DataType.FLOAT, shape=ir.Shape([3, 3, 3, 3]))

        if b_is_const:
            b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        else:
            b = ir.val("b", dtype=ir.DataType.FLOAT, shape=ir.Shape([3]))

        if scale_is_scalar:
            scale = tape.initializer(
                ir.tensor(np.array([2.0], dtype=np.float32), name="scale")
            )
        else:
            # Vector scale — should NOT be fused
            scale = tape.initializer(
                ir.tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32), name="scale")
            )

        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        conv_out = tape.op("Conv", [x, w, b], attributes={"pads": list(pads)})
        mul_out = tape.op("Mul", [conv_out, scale])
        z = tape.op(
            "Add",
            [mul_out, offset],
            output=ir.val("z", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32])),
        )

        inputs = (
            [x]
            if w_is_const and b_is_const
            else [x, w, b]
            if not w_is_const and not b_is_const
            else [x, w]
            if not w_is_const
            else [x, b]
        )
        return ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[z],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 17},
            ),
            ir_version=8,
        )

    def _build_affine_conv_model(self, *, pads: tuple = (0, 0, 0, 0)) -> ir.Model:
        """Build Mul(scale) → Add(offset) → Conv model (pre-conv affine)."""
        tape = ir.tape.Tape()
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32]))
        w = tape.initializer(ir.tensor(np.ones((3, 3, 3, 3), dtype=np.float32), name="w"))
        b = tape.initializer(ir.tensor(np.ones((3,), dtype=np.float32), name="b"))
        scale = tape.initializer(ir.tensor(np.array([2.0], dtype=np.float32), name="scale"))
        offset = tape.initializer(ir.tensor(np.array([3.0], dtype=np.float32), name="offset"))

        mul_out = tape.op("Mul", [x, scale])
        add_out = tape.op("Add", [mul_out, offset])
        conv_out = tape.op(
            "Conv",
            [add_out, w, b],
            attributes={"pads": list(pads)},
            output=ir.val("z", dtype=ir.DataType.FLOAT, shape=ir.Shape([1, 3, 32, 32])),
        )

        return ir.Model(
            ir.Graph(
                inputs=[x],
                outputs=[conv_out],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 17},
            ),
            ir_version=8,
        )

    def _clone(self, model: ir.Model) -> ir.Model:
        return ir.from_proto(ir.to_proto(model))

    # --- Negative: non-constant weight ---

    def test_conv_affine_non_constant_weight_no_fusion(self):
        """Non-constant weight → check rejects (w must be constant)."""
        model = self._build_conv_affine_model(w_is_const=False)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        # Should NOT have fused — node count unchanged
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: non-scalar scale ---

    def test_conv_affine_non_scalar_scale_no_fusion(self):
        """Vector scale → check rejects (scale must be scalar)."""
        model = self._build_conv_affine_model(scale_is_scalar=False)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: non-constant bias ---

    def test_conv_affine_non_constant_bias_no_fusion(self):
        """Non-constant bias → check rejects (b must be constant)."""
        model = self._build_conv_affine_model(b_is_const=False)
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[conv_affine_fusion_rule])
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())

    # --- Negative: pre-conv affine with padding ---

    def test_affine_conv_with_padding_no_fusion(self):
        """Pre-conv affine + padded Conv → AffineConvFusion must NOT match.

        AffineConvFusion pattern requires pads=[0,0,0,0].
        """
        model = self._build_affine_conv_model(pads=(1, 1, 1, 1))
        rewritten = self._clone(model)
        rewritten = rewrite(rewritten, pattern_rewrite_rules=[affine_conv_fusion_rule])
        # Pattern won't match — node count unchanged
        self.assertEqual(model.graph.num_nodes(), rewritten.graph.num_nodes())


if __name__ == "__main__":
    unittest.main()
