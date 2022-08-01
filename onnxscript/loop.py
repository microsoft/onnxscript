# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np


def conditional_range(max_iter, condition):
    """
    Enumerates all integer up to *max_iter* if
    *condition* is True, stops otherwise.
    *max_iter* and *condition* can be None but not both of them.
    """
    if max_iter is None and condition is None:
        raise ValueError("max_iter or/and condition must be not None.")
    if condition is None:
        for i in range(max_iter):
            yield i
    else:
        if not isinstance(condition, np.ndarray):
            raise TypeError(
                f"condition must be of type LoopCondition to enable eager mode "
                f"not {type(condition)}.")
        counter = 0
        while condition and (max_iter is None or counter < max_iter):
            yield counter
            counter += 1
