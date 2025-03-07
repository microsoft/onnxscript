# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Polyfill for Python builtin functions."""

import sys
from typing import Any, Sequence

if sys.version_info >= (3, 10):
    zip = zip  # pylint: disable=self-assigning-variable
else:
    # zip(..., strict=True) was added in Python 3.10
    # TODO: Remove this polyfill when we drop support for Python 3.9
    _python_zip = zip

    def zip(a: Sequence[Any], b: Sequence[Any], strict: bool = False):
        """Polyfill for Python's zip function.

        This is a special version which only supports two Sequence inputs.

        Raises:
            ValueError: If the iterables have different lengths and strict is True.
        """
        if len(a) != len(b) and strict:
            raise ValueError("zip() argument lengths must be equal")
        return _python_zip(a, b)
