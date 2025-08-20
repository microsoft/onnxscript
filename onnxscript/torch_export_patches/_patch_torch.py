from typing import Sequence

import torch


def patched_infer_size(a, b):
    """Patches ``torch._subclasses.fake_impls.infer_size``."""
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    dimsa = len(a)
    dimsb = len(b)
    ndim = max(dimsa, dimsb)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dima = dimsa - 1 - offset
        dimb = dimsb - 1 - offset
        sizeA = a[dima] if dima >= 0 else 1
        sizeB = b[dimb] if dimb >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = guard_size_oblivious(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = guard_size_oblivious(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = guard_size_oblivious(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
        else:
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model SmolLM.
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


def patched__broadcast_shapes(*_shapes):
    """Patches ``torch._refs._broadcast_shapes``."""
    from functools import reduce

    from torch._prims_common import IntLike
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape = [  # List[Union[int, torch.SymInt]]
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])

    return common_shape


def patched_vmap(func, in_dims=0, out_dims=0):
    """Python implementation of :func:`torch.vmap`.

    The implementation raises an issue when it is being exported with
    :func:`torch.export.export` when the function is called with
    non tensors arguments and the batch size is dynamic.
    """

    def wrapped(*args):
        in_dims_ = (
            ([in_dims] * len(args))
            if not isinstance(in_dims, (list, tuple))
            else list(in_dims)
        )
        batch_size = None
        batched_args = []
        for arg, in_dim in zip(args, in_dims_):
            if in_dim is None:
                batched_args.append(arg)
                continue

            assert batch_size is None or batch_size == arg.size(in_dim), (
                f"Unable to continue, batch_size={batch_size}, in_dim={in_dim}, "
                f"arg.size(in_dim)={arg.size(in_dim)}"
            )
            if batch_size is None:
                batch_size = arg.size(in_dim)
            arg = arg.movedim(in_dim, 0)
            batched_args.append(arg)

        if all(isinstance(a, torch.Tensor) for a in args) and isinstance(
            batch_size, torch.SymInt
        ):
            batched_tensors = [
                (
                    arg
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else arg.unsqueeze(0).expand((batch_size, *arg.shape))
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]
            results = torch.ops.higher_order.scan(func, [], batched_tensors, [])
            stacked = results[0]
            if out_dims != 0:
                return stacked.movedim(0, out_dims)
            return stacked

        else:
            torch._check(
                not isinstance(batch_size, torch.SymInt),
                lambda: (
                    f"patched_vmap supports dynamic batch_size only if all argument "
                    f"are tensors but types are {[type(a) for a in args]}"
                ),
            )
            batched_tensors = [
                (
                    (None, arg)
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else (arg, arg)
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]

            results = []
            for i in range(batch_size):
                input_slice = [v if v is not None else arg[i] for v, arg in batched_tensors]
                result = func(*input_slice)
                results.append(result)

            if isinstance(results[0], torch.Tensor):
                stacked = torch.stack(results)
                if out_dims != 0:
                    return stacked.movedim(0, out_dims)
                return stacked
            return results

    return wrapped
