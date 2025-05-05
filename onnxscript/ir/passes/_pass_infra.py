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
from typing import Literal, Sequence, final

__all__ = [
    "PassBase",
    "Sequential",
    "InPlacePass",
    "FunctionalPass",
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
        modified: Whether the resulting model is different from the input model.
    """

    model: ir.Model
    modified: bool


class PassBase(abc.ABC):
    """Base class for all passes.


    ``in_place`` and ``changes_input`` properties and what they mean:

    +------------+------------------+----------------------------+
    |            | changes_inputs   | not changes_inputs         |
    +------------+------------------+----------------------------+
    | in_place   | in place         | Side-effect-only pass      |
    +------------+------------------+----------------------------+
    | not        | destructive      | functional                 |
    | in_place   |                  |                            |
    +------------+------------------+----------------------------+
    """

    @property
    @abc.abstractmethod
    def in_place(self) -> bool:
        """Whether the pass modifies the model in place and returns it.

        If True, the pass will return the same model object that was passed in.
        If False, the pass will return a new model object.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def changes_input(self) -> bool:
        """Whether the pass modifies input model."""
        raise NotImplementedError

    @property
    def destructive(self) -> bool:
        """Whether the pass will destroy the input model when ``in_place=False``.

        A pass is destructive if it is not in place and it modifies the input model.
        """
        return not self.in_place and self.changes_input

    def __call__(self, model_or_result: ir.Model | PassResult, /) -> PassResult:
        if isinstance(model_or_result, PassResult):
            model = model_or_result.model
        else:
            model = model_or_result
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
                f"The result of the pass '{self.__class__.__name__}' should be type PassResult. "
                "Please create one with ir.passes.PassResult()."
            )

        # Checks that the declared in-place property is respected
        if self.in_place and result.model is not model:
            raise PassError(
                f"The pass '{self.__class__.__name__}' is declared in-place, "
                "but the model returned is *not* the same object as the input model. "
                "Pass developer: Pass should return the same model object or the in_place property should return False."
            )
        if not self.in_place and result.model is model:
            raise PassError(
                f"The pass '{self.__class__.__name__}' is declared not in-place, "
                "but the model returned *is* the same object as the input model. "
                "Pass developer: Pass should return a new model object or the in_place property should return True."
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


class InPlacePass(PassBase):
    """A pass that modifies the input model in place and returns it."""

    @property
    @final
    def in_place(self) -> Literal[True]:
        """An in-place pass is in place."""
        return True

    @property
    @final
    def changes_input(self) -> Literal[True]:
        """An in-place pass changes the input model."""
        return True


class FunctionalPass(PassBase):
    """A pass that returns a new model but does not modify the input model."""

    @property
    @final
    def in_place(self) -> Literal[False]:
        """A functional pass is not in place."""
        return False

    @property
    @final
    def changes_input(self) -> Literal[False]:
        """A functional pass does not change the input model."""
        return False


class Sequential(PassBase):
    """Run a sequence of passes in order."""

    def __init__(self, *passes: PassBase):
        if not passes:
            raise ValueError("Sequential must take at least one pass")
        self.passes = passes
        self._in_place = all(pass_.in_place for pass_ in passes)
        # The reason changes_inputs is decided by the first pass is that if the first pass is either in-place,
        # or if it is not designed to be in-place but somehow changes the input (destructive),
        # this pass sequence will change inputs.
        self._changes_input = self.passes[0].changes_input or self.passes[0].in_place

    @property
    def in_place(self) -> bool:
        return self._in_place

    @property
    def changes_input(self) -> bool:
        return self._changes_input

    def call(self, model: ir.Model) -> PassResult:
        modified = False
        for i, pass_ in enumerate(self.passes):
            logger.debug("Running the %s-th pass '%s'", i, pass_)
            try:
                pass_result = pass_(model)
            except Exception as e:
                prev_pass_names = [str(p) for p in self.passes[:i]]
                raise PassError(
                    f"An error occurred when running the '{pass_}' pass after the "
                    f"following passes: {prev_pass_names}"
                ) from e

            model = pass_result.model
            modified = modified or pass_result.modified

        return PassResult(model, modified)


class PassManager(Sequential):
    """Pass manager for the IR.

    The PassManager is a Pass that runs a sequence of passes on a model.

    Attributes:
        passes: The passes to run.
        steps: The number of times to run the passes.
        early_stop: Whether to stop running the passes if the graph stops changing.
    """

    def __init__(
        self,
        passes: Sequence[PassBase],
        steps: int = 1,
        early_stop: bool = True,
    ):
        # TODO(justinchuby): Implement constraints
        super().__init__(*passes)
        self.steps = steps
        self.early_stop = early_stop

    def call(self, model: ir.Model) -> PassResult:
        """Run the set of passes `steps` number of times or until the graph stops changing."""
        overall_modified = False
        for step in range(self.steps):
            try:
                # Call the call method of Sequential
                step_result = super().call(model)
            except Exception as e:
                raise PassError(f"An error occurred at step {step}") from e
            model = step_result.model
            modified = step_result.modified
            overall_modified = overall_modified or modified
            # If the graph no longer changes, then we can stop running these passes
            if not modified and self.early_stop:
                logger.info("PassManager: No more graph changes detected after step %s", step)
                break
        return PassResult(model, overall_modified)
