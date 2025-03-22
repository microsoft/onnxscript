# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# This module implements some APIs described in
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
from typing import Sequence

__all__ = [
    "PassBase",
    "PassManager",
    "PassResult",
    # Errors
    "InvariantError",
    "PreconditionError",
    "PostconditionError",
    "PassError",
]

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
    """Base class for all passes."""

    @property
    def in_place(self) -> bool:
        """Whether the pass modifies the model in place."""
        return True

    @property
    def destructive(self) -> bool:
        """Whether the pass will destroy the input model when ``in_place=False``."""
        return False

    def __call__(self, model: ir.Model) -> PassResult:
        # Check preconditions
        try:
            self.requires(model)
        except PreconditionError:
            raise
        except Exception as e:
            raise PreconditionError(
                f"Pre-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        result = self.call(model)

        # Check postconditions
        try:
            self.ensures(model)
        except PostconditionError:
            raise
        except Exception as e:
            raise PostconditionError(
                f"Post-condition for pass '{self.__class__.__name__}' failed"
            ) from e

        if not isinstance(result, PassResult):
            raise TypeError(
                f"The result of the pass '{self.__class__.__name__}' should be type PassResult."
                "Please create one with ir.passes.PassResult()."
            )
        return result

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


class PassManager(PassBase):
    """Pass manager for the IR.

    The PassManager is a Pass that runs a sequence of passes on a model.

    Attributes:
        passes: The passes to run.
        steps: The number of times to run the passes.
    """

    def __init__(
        self,
        passes: Sequence[PassBase],
        steps: int = 1,
    ):
        # TODO(justinchuby): Implement constraints
        self.passes = list(passes)
        self.steps = steps

    @property
    def in_place(self) -> bool:
        """Whether the pass modifies the model in place."""
        return all(pass_.in_place for pass_ in self.passes)

    @property
    def destructive(self) -> bool:
        """Whether the pass will destroy the input model when ``in_place=False``."""
        # This logic is a little conservative, but it is ok for now
        return any(pass_.destructive for pass_ in self.passes)

    def call(self, model: ir.Model) -> PassResult:
        """Run the set of passes `steps` number of times or until the graph stops changing."""
        overall_modified = False
        for step in range(self.steps):
            step_result = self._run_one_step(model, step)
            model = step_result.model
            modified = step_result.modified
            overall_modified = overall_modified or modified
            # If the graph no longer changes, then we can stop running these passes
            if not modified:
                logger.info("PassManager: No more graph changes detected after step %s", step)
                break
        return PassResult(model, overall_modified)

    def _run_one_step(self, model: ir.Model, step: int) -> PassResult:
        modified = False
        for i, pass_ in enumerate(self.passes):
            logger.debug("Running the %s-th pass '%s', (step %s)", i, pass_, step)
            try:
                pass_result = pass_(model)
            except (PreconditionError, PostconditionError):
                raise
            except Exception as e:
                prev_pass_names = [str(p) for p in self.passes[:i]]
                raise PassError(
                    f"An error occurred when running the '{pass_}' pass after the "
                    f"following passes: {prev_pass_names} during step {step}"
                ) from e
            if not isinstance(pass_result, PassResult):
                raise TypeError(
                    f"The result of the pass {pass_} should be type PassResult."
                    "Please create one with ir.passes.PassResult()."
                )

            model = pass_result.model
            modified = modified or pass_result.modified

        return PassResult(model, modified)
