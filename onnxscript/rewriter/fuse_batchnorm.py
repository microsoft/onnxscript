# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fuses BatchNormalization nodes into preceding nodes. Supported fusion patterns:
- BatchNormalization ∘ Conv         -> Conv
- BatchNormalization ∘ ConvTranpose -> ConvTranpose
- BatchNormalization ∘ Gemm         -> Gemm

Approach:
    Given an inbound operation output: Y = W * X + B
    And a BatchNormalization outputs: Y_BN = (gamma * (Y - μ) / std) + β, where std = sqrt(var + eps)

    The fusion updates the inbound weights as follows:
        - W_fused = W * (gamma / std)
        - B_fused = (B - μ) * (gamma / std) + β
"""

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from onnxscript import ir
from onnxscript.rewriter import pattern as orp


def _reshape_for_broadcast(x: np.ndarray, rank: int, axis: int = 1) -> np.ndarray:
    # Build shape: 1s everywhere except -1 at the target axis
    broadcast_shape = [1 if axis != i else -1 for i in range(rank)]
    return np.reshape(x, broadcast_shape)


class _FuseBatchNormBase(orp.RewriteRuleClassBase, ABC):
    """Interface for BatchNormalization nodes fusion."""

    def __init__(
        self,
        op_type: str,
        name: str | None = None,
        remove_nodes: bool = True,
        as_function: bool = False,
    ) -> None:
        super().__init__(name=name, remove_nodes=remove_nodes, as_function=as_function)
        self.op_type = op_type

    @abstractmethod
    def get_filters_axis(self, attributes: Mapping[str, ir.Attr]) -> int:
        """Return the axis along which BatchNorm scale should be broadcasted."""

    def rewrite(self, op, x: ir.Value, inbound_out: ir.Value, batchnorm_out: ir.Value):
        batchnorm_node = batchnorm_out.producer()
        # Get BatchNorm parameters
        gamma, beta, input_mean, input_var = [
            inp.const_value.numpy() for inp in batchnorm_node.inputs[1:]
        ]

        # 1e-5 is the default value for epsilon according to
        # https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#attributes
        default_eps = ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5)
        eps = batchnorm_node.attributes.get("epsilon", default_eps).as_float()

        # Compute the scale_factor to update the inbound weights and bias
        scale_factor = gamma / np.sqrt(input_var + eps)

        # Update inbound weights
        inbound_node = inbound_out.producer()
        weights = inbound_node.inputs[1].const_value.numpy()

        # Reshape scale factor so it is broadcastable
        axis = self.get_filters_axis(inbound_node.attributes)
        fused_weights = ir.tensor(
            weights * _reshape_for_broadcast(scale_factor, weights.ndim, axis=axis)
        )

        # Update bias
        if len(inbound_node.inputs) > 2:
            original_bias = inbound_node.inputs[2].const_value.numpy()
            bias_name = inbound_node.inputs[2].name
        else:
            original_bias = np.zeros_like(input_mean)
            bias_name = x.name + "_bias"
        fused_bias = ir.tensor((original_bias - input_mean) * scale_factor + beta)

        return op.op(
            self.op_type,
            inputs=[
                x,
                op.initializer(fused_weights, name=inbound_node.inputs[1].name),
                op.initializer(fused_bias, name=bias_name),
            ],
            attributes=inbound_node.attributes,
        )

    def check(
        self, context, x, inbound_out: ir.Value, batchnorm_out: ir.Value
    ) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()

        inbound_node = inbound_out.producer()
        batchnorm_node = batchnorm_out.producer()

        # Check that inbound weights + (inbound bias) + batchnorm params are initializers
        # and that they are not graph inputs
        initializers = [inbound_node.inputs[1], *batchnorm_node.inputs[1:]]
        if len(inbound_node.inputs) > 2:
            initializers.append(inbound_node.inputs[2])

        for initializer in initializers:
            if not initializer.is_initializer() or initializer.const_value is None:
                return check_result.fail(f"{initializer.name} is not a constant initializer.")
            if initializer.is_graph_input():
                return check_result.fail(f"{initializer.name} is a graph input.")

        return check_result


class FuseBatchNormIntoConv(_FuseBatchNormBase):
    """Replaces ``BatchNormalization(Conv(x))`` with ``Conv(x)``."""

    def __init__(self):
        super().__init__("Conv")

    def get_filters_axis(self, attributes: Mapping[str, ir.Attr]) -> int:
        return 0

    def pattern(self, op, x):
        return op.BatchNormalization(
            op.Conv(x, _allow_other_inputs=True, _outputs=["inbound_out"]),
            _allow_other_inputs=True,
            _outputs=["batchnorm_out"],
        )


class FuseBatchNormIntoConvTranspose(_FuseBatchNormBase):
    """Replaces ``BatchNormalization(ConvTranspose(x))`` with ``ConvTranspose(x)``."""

    def __init__(self):
        super().__init__("ConvTranspose")

    def get_filters_axis(self, attributes: Mapping[str, ir.Attr]) -> int:
        return 1

    def pattern(self, op, x):
        return op.BatchNormalization(
            op.ConvTranspose(x, _allow_other_inputs=True, _outputs=["inbound_out"]),
            _allow_other_inputs=True,
            _outputs=["batchnorm_out"],
        )


class FuseBatchNormIntoGemm(_FuseBatchNormBase):
    """Replaces ``BatchNormalization(Gemm(x))`` with ``Gemm(x)``."""

    def __init__(self):
        super().__init__("Gemm")

    def get_filters_axis(self, attributes: Mapping[str, ir.Attr]) -> int:
        return (
            0 if attributes.get("transB") is not None and attributes["transB"].as_int() else 1
        )

    def pattern(self, op, x):
        return op.BatchNormalization(
            op.Gemm(x, _allow_other_inputs=True, _outputs=["inbound_out"]),
            _allow_other_inputs=True,
            _outputs=["batchnorm_out"],
        )


fuse_batchnorm_into_conv_rule = FuseBatchNormIntoConv().rule()
fuse_batchnorm_into_convtranspose_rule = FuseBatchNormIntoConvTranspose().rule()
fuse_batchnorm_into_gemm_rule = FuseBatchNormIntoGemm().rule()


def fuse_batchnorm_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rewrite rules that fuse BatchNormalization nodes
    into preceding nodes such as Conv, ConvTranspose, and Gemm.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            fuse_batchnorm_into_conv_rule,
            fuse_batchnorm_into_convtranspose_rule,
            fuse_batchnorm_into_gemm_rule,
        ]
    )
