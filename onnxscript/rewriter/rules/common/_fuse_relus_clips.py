# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Does the following transformation:
- Relu(Relu(X)) -> Relu
- Relu(Clip(X)) -> Clip
- Clip(Relu(X)) -> Clip
- Clip(Clip(X)) -> Clip
"""

from __future__ import annotations

import abc

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class FuseSuccessiveRelu(RewriteRuleClassBase):
    """Replaces ``Relu(Relu(X))`` with ``Relu(X)``."""

    def rewrite(self, op, x):
        return op.Relu(x)

    def pattern(self, op, x):
        return op.Relu(op.Relu(x))


class _FuseReluClipBase(RewriteRuleClassBase, abc.ABC):
    def rewrite(self, op, x, **kwargs):
        first_clip_node = kwargs.get("out_first_clip").producer()
        second_clip_node = None

        if out_second_clip := kwargs.get("out_second_clip"):
            second_clip_node = out_second_clip.producer()

        min_clip, max_clip = self.compute_clip_min_max(first_clip_node, second_clip_node)
        clip_min_max = []

        if min_clip is not None:
            clip_min_max.append(
                op.initializer(min_clip, name=f"{first_clip_node.inputs[0].name}_min")
            )

        if max_clip is not None:
            # ONNX Clip expects min and max inputs in order.
            # If min is not provided, we insert None to maintain correct argument positions.
            if min_clip is None:
                clip_min_max.append(None)

            clip_min_max.append(
                op.initializer(max_clip, name=f"{first_clip_node.inputs[0].name}_max")
            )

        return op.Clip(x, *clip_min_max)

    @abc.abstractmethod
    def compute_clip_min_max(
        self, first_clip_node: ir.Node, second_clip_node: ir.Node | None = None
    ):
        pass

    def extract_min_max(self, node: ir.Node):
        # Infer dtype from node first input
        dtype = node.inputs[0].dtype.numpy()
        min_clip, max_clip = None, None

        if len(node.inputs) > 1:
            min_input = node.inputs[1]
            # If only a max is provided, min is implicitly None, so we check that
            if min_input is not None:
                min_clip = min_input.const_value.numpy()

        if len(node.inputs) > 2:
            max_clip = node.inputs[2].const_value.numpy()

        return min_clip, max_clip, dtype

    def check(self, context, **kwargs):
        """Condition to check if we need to replace the pattern.

        The pattern is applied only when the min and max inputs of the Clip nodes are
        not graph inputs and are constant values (i.e., provided by Constant nodes or initializers).

        Returns:
            MatchResult:
                Success if we need to replace the pattern, Failure otherwise.
        """
        del context  # Unused
        check_result = MatchResult()

        # Check if Clip min/max are not graph inputs and are constant values
        clip_min_max = []

        first_clip_node = kwargs.get("out_first_clip").producer()
        clip_min_max.extend([inp for inp in first_clip_node.inputs[1:] if inp is not None])

        if out_second_clip := kwargs.get("out_second_clip"):
            second_clip_node = out_second_clip.producer()
            clip_min_max.extend(
                [inp for inp in second_clip_node.inputs[1:] if inp is not None]
            )

        for m in clip_min_max:
            if m.is_graph_input():
                return check_result.fail(f"{m.name} is a graph input.")

            if ir.convenience.get_const_tensor(m) is None:
                return check_result.fail(f"{m.name} is not a constant.")

        return check_result


class FuseSuccessiveClip(_FuseReluClipBase):
    """Replaces ``Clip(Clip(X))`` with ``Clip(X)``."""

    def pattern(self, op, x):
        return op.Clip(
            op.Clip(x, _allow_other_inputs=True, _outputs=["out_first_clip"]),
            _allow_other_inputs=True,
            _outputs=["out_second_clip"],
        )

    def compute_clip_min_max(self, first_clip_node: ir.Node, second_clip_node: ir.Node):
        min_clip1, max_clip1, dtype = self.extract_min_max(first_clip_node)
        min_clip2, max_clip2, _ = self.extract_min_max(second_clip_node)

        def combine(val1, val2, op):
            if val1 is not None and val2 is not None:
                return ir.tensor(np.array(op(val1, val2), dtype=dtype))
            elif val1 is not None:
                return ir.tensor(val1)
            elif val2 is not None:
                return ir.tensor(val2)
            return None

        min_clip = combine(min_clip1, min_clip2, np.maximum)
        max_clip = combine(max_clip1, max_clip2, np.minimum)

        return min_clip, max_clip


class FuseSuccessiveClipRelu(_FuseReluClipBase):
    """Replaces ``Clip(Relu(X))`` with ``Clip(X)``."""

    def pattern(self, op, x):
        return op.Clip(op.Relu(x), _allow_other_inputs=True, _outputs=["out_first_clip"])

    def compute_clip_min_max(self, first_clip_node: ir.Node, _):
        min_clip, max_clip, dtype = self.extract_min_max(first_clip_node)

        if min_clip is None:
            # The minimum clipping value is implicitly 0 (Relu clamps at 0)
            min_clip = 0

        min_clip = ir.tensor(np.array(np.maximum(0.0, min_clip), dtype=dtype))

        if max_clip is not None:
            max_clip = ir.tensor(max_clip)
        return min_clip, max_clip


class FuseSuccessiveReluClip(FuseSuccessiveClipRelu):
    """Replaces ``Relu(Clip(X))`` with ``Clip(X)``."""

    def pattern(self, op, x):
        return op.Relu(op.Clip(x, _allow_other_inputs=True, _outputs=["out_first_clip"]))


successive_relu_rule = FuseSuccessiveRelu().rule()
successive_clip_rule = FuseSuccessiveClip().rule()
successive_clip_relu_rule = FuseSuccessiveClipRelu().rule()
successive_relu_clip_rule = FuseSuccessiveReluClip().rule()


rules = RewriteRuleSet(
    [
        successive_clip_relu_rule,
        successive_relu_clip_rule,
        successive_relu_rule,
        successive_clip_rule,
    ]
)
