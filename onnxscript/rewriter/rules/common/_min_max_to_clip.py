# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fuses successive Min/Max patterns in ONNX graphs.

Supported transformations:
- Min(Min(X, c1, c2, ...), d1, d2, ...) → Min(X, fused_const)
- Max(Max(X, c1, c2, ...), d1, d2, ...) → Max(X, fused_const)
- Min(Max(X, lb1, lb2, ...), ub1, ub2, ...) → Clip(X, lb, ub)
- Max(Min(X, ub1, ub2, ...), lb1, lb2, ...) → Clip(X, lb, ub)

Where:
    - fused_const is the reduction (min or max) over all constant inputs.
    - For Clip fusion:
        * All constant inputs must be scalars.
        * The effective lower bound is the maximum of all lower-bound constants.
        * The effective upper bound is the minimum of all upper-bound constants.

        For the case of Max(Min(X, upper_bound), lower_bound):
            * The rule applies only if lower_bound ≤ upper_bound.

General constraints:
    - The first input may be any tensor.
    - All other inputs must be constant tensors (from Constant nodes or initializers).
"""

import abc
import functools
from typing import ClassVar

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _FuseMinMaxBase(RewriteRuleClassBase, abc.ABC):
    """Base class for Min/Max fusion rewrites.

    Constraints:
        - All inputs except the first must be constants (from Constant nodes or initializers).
        - If ``need_scalars`` is True (Clip fusion), all constants must be scalars.
        - If ``check_bounds`` is True (Clip fusion in the pattern Max(Min(X, upper_bound), lower_bound)), lower_bound ≤ upper_bound.
    """

    need_scalars: ClassVar = False
    check_bounds: ClassVar = False

    @abc.abstractmethod
    def compute_constants(
        self,
        first_node: ir.Node,
        second_node: ir.Node,
        input_name: str = "",
    ) -> list[tuple[ir.Tensor, str]]: ...

    def rewrite(self, op, x, out1, out2):
        first_node = out1.producer()
        second_node = out2.producer()

        # Compute new constants for the fused op
        constants = self.compute_constants(first_node, second_node, x.name)

        initializers = [op.initializer(constant, name=name) for constant, name in constants]

        return op.op(
            self.op_type,
            inputs=[x, *initializers],
        )

    def _is_scalar(self, v: np.ndarray) -> bool:
        return np.isscalar(v) or np.size(v) == 1

    def check(self, context, out1, out2, **_):
        """Condition to check if we need to replace the pattern.

        Conditions:
            - The min and max input nodes must not be graph inputs.
            - These inputs (except the first) must be constant values (from Constant nodes or initializers).
            - In the case of Min(Max) and Max(Min) patterns:
                * All inputs must be scalars (as Clip requires scalars).
                For Max(Min) pattern:
                    * The lower bound must be less than or equal to the upper bound.

        Returns:
            MatchResult:
                Success if we need to replace the pattern, Failure otherwise.
        """
        del context  # Not used
        check_result = MatchResult()

        first_node = out1.producer()
        second_node = out2.producer()

        # Ensure all inputs except the first are constants
        for input_ in first_node.inputs[1:] + second_node.inputs[1:]:
            if ir.convenience.get_const_tensor(input_) is None:
                return check_result.fail(f"{input_.name} is not a constant.")

            # If scalars are required (Clip fusion), enforce scalar-ness
            if self.need_scalars and not self._is_scalar(input_.const_value.numpy()):
                return check_result.fail(f"{input_.name} is not a scalar.")

        if self.need_scalars and self.check_bounds:
            # For Clip fusion in the case of Max(Min(X, upper_bound), lower_bound): check that lower_bound <= upper_bound
            lower_bound, upper_bound = self.compute_constants(first_node, second_node)
            if lower_bound[0].numpy() > upper_bound[0].numpy():
                return check_result.fail(
                    f"Invalid bounds: lower bound ({lower_bound[0].numpy()}) is greater "
                    f"than upper bound ({upper_bound[0].numpy()})."
                )

        return check_result


class FuseSuccessiveMin(_FuseMinMaxBase):
    """Replaces ``Min(Min(X, c1, c2, ...), d1, d2, ...)`` with ``Min(X, fused_const)``.

    Constraints:
        - All inputs except the first must be constants (from Constant nodes or initializers).
    """

    op_type: ClassVar = "Min"

    def compute_constants(
        self,
        first_node: ir.Node,
        second_node: ir.Node,
        input_name: str = "",
    ) -> list[tuple[ir.Tensor, str]]:
        inputs = first_node.inputs[1:] + second_node.inputs[1:]
        values = [input_.const_value.numpy() for input_ in inputs]
        return [(ir.tensor(functools.reduce(np.minimum, values)), f"{input_name}_min")]

    def pattern(self, op, x):
        return op.Min(
            op.Min(x, _allow_other_inputs=True, _outputs=["out1"]),
            _allow_other_inputs=True,
            _outputs=["out2"],
        )


class FuseSuccessiveMax(_FuseMinMaxBase):
    """Replaces ``Max(Max(X, c1, c2, ...), d1, d2, ...)`` with ``Max(X, fused_const)``.

    Constraints:
        - All inputs except the first must be constants (from Constant nodes or initializers).
    """

    op_type: ClassVar = "Max"

    def compute_constants(
        self,
        first_node: ir.Node,
        second_node: ir.Node,
        input_name: str = "",
    ) -> list[tuple[ir.Tensor, str]]:
        inputs = first_node.inputs[1:] + second_node.inputs[1:]
        values = [input_.const_value.numpy() for input_ in inputs]
        return [(ir.tensor(functools.reduce(np.maximum, values)), f"{input_name}_max")]

    def pattern(self, op, x):
        return op.Max(
            op.Max(x, _allow_other_inputs=True, _outputs=["out1"]),
            _allow_other_inputs=True,
            _outputs=["out2"],
        )


class FuseMaxMinToClip(_FuseMinMaxBase):
    """Replaces ``Min(Max(X, lb1, lb2, ...), ub1, ub2, ...)`` with ``Clip(X, lb, ub)``.

    Constraints:
        - All inputs except the first must be constants (from Constant nodes or initializers).
        - All constant inputs must be scalars.
        - The effective lower bound is ``max(lb1, lb2, ...)``.
        - The effective upper bound is ``min(ub1, ub2, ...)``.
    """

    op_type: ClassVar = "Clip"
    need_scalars: ClassVar = True

    def compute_constants(
        self,
        first_node: ir.Node,
        second_node: ir.Node,
        input_name: str = "",
    ) -> list[tuple[ir.Tensor, str]]:
        lower_bound = np.max([input_.const_value.numpy() for input_ in first_node.inputs[1:]])
        upper_bound = np.min([input_.const_value.numpy() for input_ in second_node.inputs[1:]])
        return [
            (ir.tensor(lower_bound), f"{input_name}_min"),
            (ir.tensor(upper_bound), f"{input_name}_max"),
        ]

    def pattern(self, op, x):
        return op.Min(
            op.Max(x, _allow_other_inputs=True, _outputs=["out1"]),
            _allow_other_inputs=True,
            _outputs=["out2"],
        )


class FuseMinMaxToClip(_FuseMinMaxBase):
    """Replaces ``Max(Min(X, ub1, ub2, ...), lb1, lb2, ...)`` with ``Clip(X, lb, ub)``.

    Constraints:
        - All inputs except the first must be constants (from Constant nodes or initializers).
        - All constant inputs must be scalars.
        - The effective lower bound is ``max(lb1, lb2, ...)``.
        - The effective upper bound is ``min(ub1, ub2, ...)``.
        - Requires ``lower_bound <= upper_bound``.
    """

    op_type: ClassVar = "Clip"
    need_scalars: ClassVar = True
    check_bounds: ClassVar = True

    def compute_constants(
        self,
        first_node: ir.Node,
        second_node: ir.Node,
        input_name: str = "",
    ) -> list[tuple[ir.Tensor, str]]:
        upper_bound = np.min([input_.const_value.numpy() for input_ in first_node.inputs[1:]])
        lower_bound = np.max([input_.const_value.numpy() for input_ in second_node.inputs[1:]])
        return [
            (ir.tensor(lower_bound), f"{input_name}_min"),
            (ir.tensor(upper_bound), f"{input_name}_max"),
        ]

    def pattern(self, op, x):
        return op.Max(
            op.Min(x, _allow_other_inputs=True, _outputs=["out1"]),
            _allow_other_inputs=True,
            _outputs=["out2"],
        )


min_min_rule = FuseSuccessiveMin().rule()
max_max_rule = FuseSuccessiveMax().rule()
min_max_rule = FuseMinMaxToClip().rule()
max_min_rule = FuseMaxMinToClip().rule()


rules = RewriteRuleSet(
    [
        min_min_rule,
        max_max_rule,
        min_max_rule,
        max_min_rule,
    ]
)
