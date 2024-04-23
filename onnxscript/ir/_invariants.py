# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utilities to enforce invariants on the IR."""

from __future__ import annotations

import functools
from typing import Any, Callable


class InvariantError(Exception):
    """Raised when an invariant is violated."""


class PreconditionError(InvariantError):
    """Raised when a precondition is violated."""


class PostconditionError(InvariantError):
    """Raised when a postcondition is violated."""


def requires(
    preconditions: Callable[..., str | None],
) -> Callable[..., Callable[..., Any]]:
    """Decorator to enforce preconditions on a function."""
    # TODO(justinchuby): Preserve python function signature with this decorator

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            message = preconditions(*args, **kwargs)
            if message is not None:
                raise PreconditionError(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def ensures(
    postconditions: Callable[..., str | None],
) -> Callable[..., Callable[..., Any]]:
    """Decorator to enforce postconditions on a function."""

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            result = func(*args, **kwargs)
            message = postconditions(*args, **kwargs)
            if message is not None:
                raise PostconditionError(message)
            return result

        return wrapper

    return decorator
