# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Version utils for testing."""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import packaging.version


def onnx_older_than(version: str) -> bool:
    """Returns True if the ONNX version is older than the given version."""
    import onnx  # pylint: disable=import-outside-toplevel

    return (
        packaging.version.parse(onnx.__version__).release
        < packaging.version.parse(version).release
    )


def torch_older_than(version: str) -> bool:
    """Returns True if the torch version is older than the given version."""
    import torch  # pylint: disable=import-outside-toplevel

    return (
        packaging.version.parse(torch.__version__).release
        < packaging.version.parse(version).release
    )


def transformers_older_than(version: str) -> bool | None:
    """Returns True if the transformers version is older than the given version."""
    try:
        import transformers  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    return (
        packaging.version.parse(transformers.__version__).release
        < packaging.version.parse(version).release
    )


def is_onnxruntime_training() -> bool:
    """Returns True if the onnxruntime is onnxruntime-training."""
    try:
        from onnxruntime import training  # pylint: disable=import-outside-toplevel

        assert training
    except ImportError:
        # onnxruntime not training
        return False

    try:
        from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=import-outside-toplevel
            OrtValueVector,
        )
    except ImportError:
        return False

    return hasattr(OrtValueVector, "push_back_batch")


def onnxruntime_older_than(version: str) -> bool:
    """Returns True if the onnxruntime version is older than the given version."""
    import onnxruntime  # pylint: disable=import-outside-toplevel

    return (
        packaging.version.parse(onnxruntime.__version__).release
        < packaging.version.parse(version).release
    )


def numpy_older_than(version: str) -> bool:
    """Returns True if the numpy version is older than the given version."""
    import numpy  # pylint: disable=import-outside-toplevel

    return (
        packaging.version.parse(numpy.__version__).release
        < packaging.version.parse(version).release
    )


def has_transformers():
    """Tells if transformers is installed."""
    try:
        import transformers  # pylint: disable=import-outside-toplevel

        assert transformers
        return True  # noqa
    except ImportError:
        return False


def ignore_warnings(warns: Warning | Sequence[Warning]) -> Callable:  # type: ignore[arg-type]
    """Catches warnings.

    Args:
        warns: warnings to ignore

    Returns:
        decorated function
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)  # type: ignore[arg-type]
                return fct(self)

        return call_f

    return wrapper
