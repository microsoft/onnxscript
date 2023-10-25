"""Experimental flags.

NOTE: These flags are experimental only. Any flag here can be removed at any
time without notice.
"""

import os

EXPERIMENTAL_INITIALIZERS_AS_INPUTS = (
    os.getenv("TORCHLIB_EXPERIMENTAL_INITIALIZERS_AS_INPUTS") == "1"
)
