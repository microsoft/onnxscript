# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Experimental flags.

NOTE: These flags are experimental only. Any flag here can be removed at any
time without notice.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _load_boolean_flag(
    name: str,
    *,
    this_will: str,
    deprecated: bool = False,
    default: bool = False,
) -> bool:
    """Load a boolean flag from environment variable.

    Args:
        name: The name of the environment variable.
        this_will: A string that describes what this flag will do.
        deprecated: Whether this flag is deprecated.
        default: The default value if envvar not defined.
    """
    undefined = os.getenv(name) is None
    state = os.getenv(name) == "1"
    if state:
        if deprecated:
            logger.error(
                "Experimental flag %s is deprecated. Please remove it from your environment.",
                name,
            )
        else:
            logger.warning("Experimental flag %s is enabled. This will %s.", name, this_will)
    if undefined:
        state = default
    return state


EXPERIMENTAL_INITIALIZERS_AS_INPUTS: bool = _load_boolean_flag(
    "TORCHLIB_EXPERIMENTAL_INITIALIZERS_AS_INPUTS",
    this_will="make initializers as inputs to the model graph",
)
EXPERIMENTAL_PREFER_TRACING: bool = _load_boolean_flag(
    "TORCHLIB_EXPERIMENTAL_PREFER_TRACING",
    this_will="trace all traceable functions to fold if branches and collapse constant expressions",
    default=True,
)
EXPERIMENTAL_USE_IR: bool = _load_boolean_flag(
    "TORCHLIB_EXPERIMENTAL_USE_IR",
    this_will="use the ONNX IR instead of the PyTorch Graph for graph building",
)
