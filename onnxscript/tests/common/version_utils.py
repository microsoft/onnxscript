"""Version utils for testing."""

import onnx
import packaging.version
import torch


def onnx_older_than(version: str) -> bool:
    """Returns True if the ONNX version is older than the given version."""
    return (
        packaging.version.parse(onnx.__version__).release
        < packaging.version.parse(version).release
    )


def torch_older_than(version: str) -> bool:
    """Returns True if the torch version is older than the given version."""
    return (
        packaging.version.parse(torch.__version__).release
        < packaging.version.parse(version).release
    )
