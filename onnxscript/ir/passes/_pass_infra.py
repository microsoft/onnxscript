# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This module implements NodeTransformer API described in
# https://pytorch.org/executorch/stable/compiler-custom-compiler-passes.html
# for the ONNX IR.
# The classes {PassResult and PassManager} are derived from
# https://github.com/pytorch/pytorch/blob/1e47c7b11b312b47a621efd547f5c90081f0d9cb/torch/fx/passes/infra/pass_base.py#L12
# and
# https://github.com/pytorch/pytorch/blob/1e47c7b11b312b47a621efd547f5c90081f0d9cb/torch/fx/passes/infra/pass_manager.py#L147
# The original code is licensed under the PyTorch License https://github.com/pytorch/pytorch/blob/main/LICENSE

"""Passes infrastructure for the IR."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Sequence

__all__ = ["PassBase", "NodeTransformer"]

import abc

from onnxscript import ir


logger = logging.getLogger(__name__)


class InvariantError(Exception):
    """Raised when an invariant is violated."""


class PreconditionError(InvariantError):
    """Raised when a precondition is violated."""


class PostconditionError(InvariantError):
    """Raised when a postcondition is violated."""


class PassError(RuntimeError):
    """Raised when an error occurs during a pass."""


@dataclasses.dataclass
class PassResult:
    """Result of a pass.

    Attributes:
        model: The transformed model.
        modified: Whether the model was modified.
    """

    model: ir.Model
    modified: bool

class PassBase(abc.ABC):
    """Base class for all passes.

    Class attributes:
        in_place: Whether the pass modifies the model in place.
    """

    in_place: bool = True
    modified: bool

    def __call__(self, model: ir.Model) -> PassResult:
        return self.call(model)

    @abc.abstractmethod
    def call(self, model: ir.Model) -> PassResult:
        """The main entry point for the pass."""
        ...

    def requires(self, model: ir.Model) -> None:
        """Pre-conditions for the pass.

        This is optional to implement, will be called before call() if run by a pass manager.
        """
        del model  # Unused

    def ensures(self, model: ir.Model) -> None:
        """Post-conditions for the pass.

        This is optional to implement, will be called after call() if run by a pass manager.
        """
        del model  # Unused


class NodeTransformer(PassBase):
    """NodeTransformer for the ONNX IR.

    An NodeTransformer is a pass that traverses the IR and performs some
    operation on the nodes. The operation can be anything, such as
    checking invariants, transforming the IR, or generating code.

    By default, the NodeTransformer updates the model in place.

    Attributes:
        _model: ir.Model: The model being interpreted.
        scope (list[ir.Graph]): The current graph the NodeTransformer is running on.
    """

    def __init__(self, reversed: bool = False):
        self._model: ir.Model | None = None
        self.scope: list[ir.Graph] = []
        self.reversed = reversed
        self.modified = True

    def call(self, model: ir.Model) -> PassResult:
        self._model = model
        self._call_graph(self._model.graph)
        return PassResult(self._model, self.modified)

    def _call_graph(self, graph: ir.Graph):
        self.enter_graph(graph)
        self.scope.append(graph)
        iterable = reversed(graph) if self.reversed else graph
        for node in iterable:
            self._call_node(node)
        self.exit_graph(graph)
        self.scope.pop()

    def _call_node(self, node: ir.Node):
        self.call_node(node)
        for attr in node.attributes.values():
            if not isinstance(attr, ir.Attr):
                continue
            if attr.type == ir.AttributeType.GRAPH:
                self._call_graph(attr.value)
            elif attr.type == ir.AttributeType.GRAPHS:
                for graph in attr.value:
                    self._call_graph(graph)

    def enter_graph(self, graph: ir.Graph):
        """Called when entering a graph. Optional to implement."""
        del graph  # Unused

    def exit_graph(self, graph: ir.Graph):
        """Called when exiting a graph. Optional to implement."""
        del graph  # Unused

    @abc.abstractmethod
    def call_node(self, node: ir.Node):
        """Called when visiting a node."""
        ...


class PassManager:
    """Pass manager for the IR.

    Attributes:
        passes: The passes to run.
        check_invariants: Whether to check invariants before and after each pass.
        steps: The number of times to run the passes.
    """

    def __init__(self, passes: list[Callable[[ir.Model], PassResult]], check_invariants: bool = False, steps: int = 1):
        # TODO(justinchuby): Implement constraints
        self.passes = list(passes)
        self.check_invariants = check_invariants
        self.steps = steps

    def _run_one_step(self, model: ir.Model) -> PassResult:

    def run(self, model: ir.Model) -> PassResult:
        # Run the set of passes `steps` number of times or until the graph stops changing
        overall_modified = False
        for step in range(self.steps):
            modified = False
            for i, pass_ in enumerate(self.passes):
                logger.debug("Running the %s-th pass '%s', (step %s)", i, pass_, step)

                # 1. Check preconditions
                if self.check_invariants:
                    try:
                        pass_.requires(model)
                    except Exception as e:
                        raise PreconditionError(f"Pre-condition failed for {pass_}") from e

                # 2. Run the pass
                try:
                    pass_result = pass_(model)
                except Exception as e:
                    prev_pass_names = [
                        str(p)
                        for p in self.passes[:i]
                    ]
                    raise PassError(f"An error occurred when running the '{pass_}' pass after the following passes: {prev_pass_names}") from e
                if not isinstance(pass_result, PassResult):
                    raise TypeError(
                        f"The result of the pass {pass_} should be type PassResult."
                        "Please create one with ir.passes.PassResult()."
                    )

                model = pass_result.model
                modified = modified or pass_result.modified

                # 3. Check postconditions
                if self.check_invariants:
                    try:
                        pass_.ensures(model)
                    except Exception as e:
                        raise PostconditionError(f"Post-condition failed for {pass_}") from e

            overall_modified = overall_modified or modified
            # If the graph no longer changes, then we can stop running these passes
            if not modified:
                logger.info("PassManager: No more graph changes detected after step %s", step)
                break

        return PassResult(model, overall_modified)
