# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

def conditional_range(max_iter, condition):
    """
    Enumerates all integer up to *max_iter* if
    *condition* is True, stops otherwise.
    *max_iter* and *condition* can be None but not both of them.
    """
    if max_iter is None and condition is None:
        raise RuntimeError("max_iter or/and condition must be not None.")
    counter = 0
    if (condition is None or condition) and (max_iter is None or counter < max_iter):
        yield counter
        counter += 1
