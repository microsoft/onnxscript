# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fuses Reduce{Max,Min} and Arg{Max,Min} patterns into TopK.

Supported transformations:
- ReduceMax(X, axes=[axis], keepdims=k) + ArgMax(X, axis=axis, keepdims=k) → TopK(X, k=1, axis=axis, largest=1) [+ Squeeze if k=0]
- ReduceMin(X, axes=[axis], keepdims=k) + ArgMin(X, axis=axis, keepdims=k) → TopK(X, k=1, axis=axis, largest=0) [+ Squeeze if k=0]

Constraints:
    - This rule only works for opset 18+.
    - Both nodes must operate on the same input X.
    - Both nodes must target the same axis.
    - Both nodes must have the same keepdims attribute value.
    - The Reduce node must operate on a single axis (len(axes) == 1).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

_TOPK_SUPPORTED_DTYPES = frozenset(
    {
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
        ir.DataType.UINT8,
        ir.DataType.UINT16,
        ir.DataType.UINT32,
        ir.DataType.UINT64,
    }
)


class _FuseReduceArgToTopKBase(RewriteRuleClassBase):
    """Base class for fusing Reduce{Max,Min} + Arg{Max,Min} into TopK.

    This base class contains the common logic for checking and rewriting patterns where
    a Reduce operation and its corresponding Arg operation can be replaced with a single
    TopK operation.

    Subclasses must implement:
        - pattern(): Define the specific Reduce and Arg operations to match
        - largest: Property returning 1 for Max operations, 0 for Min operations
    """

    @property
    @abstractmethod
    def largest(self) -> int:
        """Return 1 for Max operations (largest elements), 0 for Min operations (smallest elements)."""

    @staticmethod
    def _normalize_axis(axis: int, rank: int | None) -> int:
        """Normalize a potentially negative axis to a positive axis index.

        Args:
            axis: The axis to normalize (can be negative).
            rank: The rank of the tensor, or None if unknown.

        Returns:
            The normalized axis (non-negative if rank is known and axis was negative).
        """
        if rank is not None and axis < 0:
            return axis + rank
        return axis

    def check(self, context, reduce_val, arg_idx, **_) -> MatchResult:
        """Check if Reduce and Arg operations can be safely fused into TopK.

        Conditions:
            - Input dtype must be supported by TopK.
            - Both nodes must have the same keepdims attribute.
            - The Reduce node must operate on a single axis.
            - Both nodes must operate on the same axis.
            - The Arg node must not use select_last_index=1 (TopK doesn't support this).

        Args:
            context: The rewrite context (unused).
            reduce_val: The output of the Reduce operation (ReduceMax/ReduceMin).
            arg_idx: The output of the Arg operation (ArgMax/ArgMin).

        Returns:
            MatchResult: Success if the pattern can be fused, Failure otherwise.
        """
        del context  # Unused
        check_result = MatchResult()

        reduce_node = reduce_val.producer()
        arg_node = arg_idx.producer()

        # Get input tensor to check dtype and rank
        input_x = reduce_node.inputs[0]

        if input_x.dtype is not None and input_x.dtype not in _TOPK_SUPPORTED_DTYPES:
            return check_result.fail(f"Input dtype {input_x.dtype} is not supported by TopK")

        # Get keepdims attribute from both nodes
        reduce_keepdims = reduce_node.attributes.get_int("keepdims", 1)
        arg_keepdims = arg_node.attributes.get_int("keepdims", 1)

        # Check if keepdims match
        if reduce_keepdims != arg_keepdims:
            return check_result.fail(
                f"keepdims mismatch: {reduce_node.op_type} has {reduce_keepdims}, "
                f"{arg_node.op_type} has {arg_keepdims}."
            )

        # Get axes from Reduce node's inputs.
        # Opset 18+ moved axes from an attribute to a second input; pre-18 models
        # with axes-as-attribute won't reach this branch, so the coincidental guard
        # against pre-18 graphs is intentional and sufficient.
        if len(reduce_node.inputs) >= 2 and reduce_node.inputs[1] is not None:
            axes_input = reduce_node.inputs[1]
            axes_tensor = ir.convenience.get_const_tensor(axes_input)
            if axes_tensor is None:
                return check_result.fail(
                    f"{reduce_node.op_type} axes input is not a constant."
                )
            try:
                axes_array = axes_tensor.numpy()
                axes_list = axes_array.tolist() if axes_array.ndim > 0 else [int(axes_array)]
            except (ValueError, TypeError):
                return check_result.fail(f"Cannot parse {reduce_node.op_type} axes input.")
        else:
            return check_result.fail(f"{reduce_node.op_type} axes not found in inputs.")

        # Check that Reduce operates on exactly one axis
        if len(axes_list) != 1:
            return check_result.fail(
                f"{reduce_node.op_type} must operate on a single axis, got {len(axes_list)} axes."
            )

        reduce_axis = axes_list[0]

        # Get axis from Arg operation; ONNX default: axis=0 for ArgMax/ArgMin
        arg_axis = arg_node.attributes.get_int("axis", 0)

        # Check select_last_index attribute: TopK always returns the first occurrence in case of ties
        select_last_index = arg_node.attributes.get_int("select_last_index", 0)
        if select_last_index != 0:
            return check_result.fail(
                f"{arg_node.op_type} has select_last_index=1, which is not supported by TopK."
            )

        # Normalize axes if rank is known (handle negative indices)
        rank = input_x.shape.rank() if input_x.shape is not None else None

        if self._normalize_axis(reduce_axis, rank) != self._normalize_axis(arg_axis, rank):
            return check_result.fail(
                f"Axis mismatch: {reduce_node.op_type} operates on axis {reduce_axis}, "
                f"{arg_node.op_type} operates on axis {arg_axis}."
            )

        return check_result

    def rewrite(self, op, x, reduce_val, arg_idx):
        """Rewrite the matched pattern with TopK (and optionally Squeeze).

        Args:
            op: The operation builder.
            x: The input to both Reduce and Arg operations.
            reduce_val: The output of the Reduce operation.
            arg_idx: The output of the Arg operation.

        Returns:
            Tuple of (values, indices) matching the original outputs.
        """
        # Get the nodes
        arg_node = arg_idx.producer()

        # Extract necessary attributes with ONNX default values
        axis = arg_node.attributes.get_int("axis", 0)
        keepdims = arg_node.attributes.get_int("keepdims", 1)

        # Normalize axis (convert negative to positive) if rank is known
        axis = self._normalize_axis(axis, x.shape.rank() if x.shape is not None else None)

        # Create K constant
        k_constant = op.Constant(value=ir.tensor(np.array([1], dtype=np.int64)))

        # Create TopK node
        topk_values, topk_indices = op.TopK(
            x,
            k_constant,
            axis=axis,
            largest=self.largest,
            _outputs=2,
        )

        # Handle keepdims=0 case: TopK always keeps the reduced dimension (size 1);
        # squeeze it to match the original Reduce/Arg output shapes when keepdims=0.
        if keepdims == 0:
            axes_constant = op.Constant(value=ir.tensor(np.array([axis], dtype=np.int64)))
            new_values = op.Squeeze(topk_values, axes_constant)
            new_indices = op.Squeeze(topk_indices, axes_constant)
        else:
            new_values = topk_values
            new_indices = topk_indices

        return new_values, new_indices


class FuseReduceMaxArgMaxToTopK(_FuseReduceArgToTopKBase):
    """Replaces ReduceMax + ArgMax with TopK(largest=1).

    NOTE: Requires opset 18+. Apply a version converter before using on older models.

    Transformation:
        ReduceMax(X, axes=[axis], keepdims=k) + ArgMax(X, axis=axis, keepdims=k)
        → TopK(X, k=1, axis=axis, largest=1) [+ Squeeze if k=0]

    When keepdims=0, the output of TopK is squeezed to match the original output shapes.
    """

    @property
    def largest(self) -> int:
        return 1  # TopK returns largest elements

    def pattern(self, op, x):
        """Define the pattern to match: ReduceMax and ArgMax on the same input.

        Note: For opset 18+, ReduceMax has a second input for axes, which we allow
        but will validate in check() to ensure it's a constant.
        """
        reduce_val = op.ReduceMax(x, _allow_other_inputs=True, _outputs=["reduce_val"])
        arg_idx = op.ArgMax(x, _outputs=["arg_idx"])
        return reduce_val, arg_idx


class FuseReduceMinArgMinToTopK(_FuseReduceArgToTopKBase):
    """Replaces ReduceMin + ArgMin with TopK(largest=0).

    NOTE: Requires opset 18+. Apply a version converter before using on older models.

    Transformation:
        ReduceMin(X, axes=[axis], keepdims=k) + ArgMin(X, axis=axis, keepdims=k)
        → TopK(X, k=1, axis=axis, largest=0) [+ Squeeze if k=0]

    When keepdims=0, the output of TopK is squeezed to match the original output shapes.
    """

    @property
    def largest(self) -> int:
        return 0  # TopK returns smallest elements

    def pattern(self, op, x):
        """Define the pattern to match: ReduceMin and ArgMin on the same input.

        Note: For opset 18+, ReduceMin has a second input for axes, which we allow
        but will validate in check() to ensure it's a constant.
        """
        reduce_val = op.ReduceMin(x, _allow_other_inputs=True, _outputs=["reduce_val"])
        arg_idx = op.ArgMin(x, _outputs=["arg_idx"])
        return reduce_val, arg_idx


fuse_reduce_max_argmax_to_topk_rule = FuseReduceMaxArgMaxToTopK().rule()
fuse_reduce_min_argmin_to_topk_rule = FuseReduceMinArgMinToTopK().rule()

rules = RewriteRuleSet(
    [fuse_reduce_max_argmax_to_topk_rule, fuse_reduce_min_argmin_to_topk_rule]
)
