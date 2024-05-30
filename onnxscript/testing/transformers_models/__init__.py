# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import random


def has_transformers():
    """Tells if transformers is installed."""
    try:
        import transformers

        assert transformers
        return True  # noqa
    except ImportError:
        return False


def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    import torch

    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()
