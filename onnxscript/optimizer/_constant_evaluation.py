# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Concrete custom-op evaluation without exposing optimizer implementation state."""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping, Sequence

import onnx_ir as ir


@dataclasses.dataclass(frozen=True)
class ConstantEvaluationContext:
    """Information about the operator being evaluated.

    Evaluators must be deterministic and must not mutate inputs or attributes.
    They return one tensor per output, or None to decline evaluation. They do
    not receive a graph builder or mutable optimizer analysis state.
    """

    domain: str
    op_type: str
    version: int
    output_count: int


ConstantEvaluator = Callable[
    [
        Sequence[ir.TensorProtocol | None],
        Mapping[str, ir.Attr],
        ConstantEvaluationContext,
    ],
    Sequence[ir.TensorProtocol] | None,
]


def _normalize_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise TypeError(f"Operator domain must be a string, got {domain!r}.")
    return "" if domain == "ai.onnx" else domain


class ConstantEvaluatorRegistry:
    """An explicit set of concrete evaluators passed to fold_constants or optimize.

    Registrations are local to this object, not process-global. Overlapping
    version ranges for the same operator are rejected. Evaluation uses a snapshot
    of the registrations taken at the beginning of each optimizer pass.

    Custom evaluators run after built-in partial simplifications and the usual
    constant-input, determinism, blacklist, and size checks, but before ONNX
    reference evaluation. Returning None allows reference evaluation to proceed.
    Use the rewriter API instead for transformations of nonconstant inputs.
    """

    def __init__(self) -> None:
        self._evaluators: dict[
            tuple[str, str], list[tuple[int | None, int | None, ConstantEvaluator]]
        ] = {}

    def register(
        self,
        opname: str,
        domain: str = "",
        version: int | tuple[int | None, int | None] | None = None,
    ) -> Callable[[ConstantEvaluator], ConstantEvaluator]:
        """Register an evaluator for an operator and inclusive opset range.

        Args:
            opname: Nonempty operator name.
            domain: Operator domain; "" and "ai.onnx" are equivalent.
            version: An exact positive version, an inclusive (minimum, maximum)
                tuple with None for an unbounded endpoint, or None for all versions.

        Returns:
            A decorator preserving the callback.

        Raises:
            TypeError: An argument or callback has an unsupported type.
            ValueError: A name or version range is invalid, or ranges overlap.
        """
        if not isinstance(opname, str):
            raise TypeError(f"Operator name must be a string, got {opname!r}.")
        if not opname or not opname.strip():
            raise ValueError("Operator name must not be empty; provide an ONNX op_type.")
        domain = _normalize_domain(domain)
        if version is None:
            minimum, maximum = None, None
        elif type(version) is int:
            minimum, maximum = version, version
        elif isinstance(version, tuple) and len(version) == 2:
            minimum, maximum = version
        else:
            raise TypeError(
                f"Invalid version {version!r} for {domain}::{opname}; "
                "use a positive integer, a (minimum, maximum) tuple, or None."
            )
        for endpoint in (minimum, maximum):
            if endpoint is not None and (type(endpoint) is not int or endpoint <= 0):
                raise ValueError(
                    f"Invalid version endpoint {endpoint!r} for {domain}::{opname}; "
                    "use a positive integer or None."
                )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"Invalid version range {version!r} for {domain}::{opname}; "
                "the minimum must not exceed the maximum."
            )

        def decorator(function: ConstantEvaluator) -> ConstantEvaluator:
            if not callable(function):
                raise TypeError(
                    f"Evaluator for {domain}::{opname} must be callable, got {function!r}."
                )
            entries = self._evaluators.setdefault((domain, opname), [])
            for lower, upper, _ in entries:
                if (maximum is None or lower is None or lower <= maximum) and (
                    upper is None or minimum is None or minimum <= upper
                ):
                    raise ValueError(
                        f"An evaluator for {domain}::{opname} overlaps version range "
                        f"{version!r}; use disjoint ranges or a separate registry."
                    )
            entries.append((minimum, maximum, function))
            return function

        return decorator

    def _snapshot(self) -> ConstantEvaluatorRegistry:
        result = ConstantEvaluatorRegistry()
        result._evaluators = {key: list(entries) for key, entries in self._evaluators.items()}
        return result

    def _lookup(self, domain: str, opname: str, version: int) -> ConstantEvaluator | None:
        for minimum, maximum, evaluator in self._evaluators.get(
            (_normalize_domain(domain), opname), ()
        ):
            if (minimum is None or minimum <= version) and (
                maximum is None or version <= maximum
            ):
                return evaluator
        return None


def register_constant_folder(
    opname: str,
    domain: str = "",
    version: int | tuple[int | None, int | None] | None = None,
    *,
    registry: ConstantEvaluatorRegistry,
) -> Callable[[ConstantEvaluator], ConstantEvaluator]:
    """Register a concrete constant evaluator in an explicit registry.

    This is equivalent to ``registry.register(opname, domain, version)``.
    Pass that registry as ``evaluators=registry`` to ``fold_constants`` or
    ``optimize``. No process-global registration is performed.

    Example::

        from typing import Mapping, Sequence
        import onnx_ir as ir
        from onnxscript import optimizer

        evaluators = optimizer.ConstantEvaluatorRegistry()

        @optimizer.register_constant_folder(
            "Double", domain="example", version=1, registry=evaluators
        )
        def double(
            inputs: Sequence[ir.TensorProtocol | None],
            attributes: Mapping[str, ir.Attr],
            context: optimizer.ConstantEvaluationContext,
        ) -> Sequence[ir.TensorProtocol] | None:
            if inputs[0] is None:
                return None
            return [ir.tensor(inputs[0].numpy() * 2)]

        # optimizer.fold_constants(model, evaluators=evaluators)
    """
    return registry.register(opname, domain, version)
