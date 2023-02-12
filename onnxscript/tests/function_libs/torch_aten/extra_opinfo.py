"""
Test data for aten operators which don't exist in PyTorch file:
pytorch/torch/testing/_internal/common_methods_invocations.py.
"""

import functools
from typing import Any, List

import torch
from torch import testing as torch_testing
from torch.testing._internal import common_dtype, common_utils
from torch.testing._internal.opinfo import core as opinfo_core


def sample_inputs_conv3d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]] = (  # type: ignore[assignment]
        (
            (1, 3, 3, 224, 224),
            (32, 3, 3, 3, 3),
            None,
            {
                "stride": (2, 2, 2),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
                "groups": 1,
            },
        ),
        (
            (2, 4, 3, 56, 56),
            (32, 4, 3, 3, 3),
            (32,),
            {
                "stride": (3, 3, 3),
                "padding": 2,
                "dilation": (1, 1, 1),
                "groups": 1,
            },
        ),
    )

    for input_shape, weight, bias, kwargs in cases:  # type: ignore[assignment]
        # Batched
        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )
        # Unbatched
        yield opinfo_core.SampleInput(
            make_arg(input_shape[1:]),  # type: ignore[index]
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )


def sample_inputs_convolution(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]] = (  # type: ignore[assignment]
        (
            (1, 3, 4),
            (3, 3, 3),
            (3,),
            {
                "stride": (2,),
                "padding": (2,),
                "dilation": (1,),
                "transposed": False,
                "output_padding": (0,),
                "groups": 1,
            },
        ),
        (
            (1, 3, 4),
            (3, 3, 3),
            None,
            {
                "stride": (2,),
                "padding": (2,),
                "dilation": (1,),
                "transposed": True,
                "output_padding": (0,),
                "groups": 1,
            },
        ),
        (
            (1, 3, 224, 224),
            (32, 3, 3, 3),
            None,
            {
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": (1, 1),
                "transposed": False,
                "output_padding": (0, 0),
                "groups": 1,
            },
        ),
        (
            (1, 3, 3, 224, 224),
            (32, 3, 3, 3, 3),
            (32,),
            {
                "stride": (2, 2, 2),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
                "transposed": False,
                "output_padding": (0, 0, 0),
                "groups": 1,
            },
        ),
        # FIXME(jiz): Uncomment out these test data once
        # torch 2.0 is released.
        # (
        #     (1, 3, 224, 224, 224),
        #     (32, 3, 3, 3, 3),
        #     (32,),
        #     {
        #         "stride": (2, 2, 2),
        #         "padding": (1, 1, 1),
        #         "dilation": (1, 1, 1),
        #         "transposed": False,
        #         "output_padding": (0, 0, 0),
        #         "groups": 1,
        #     },
        # ),
        (
            (2, 4, 6, 6),
            (4, 1, 3, 3),
            (4,),
            {
                "stride": (3, 2),
                "padding": (1, 1),
                "dilation": (1, 1),
                "transposed": True,
                "output_padding": (0, 0),
                "groups": 4,
            },
        ),
    )

    for input_shape, weight, bias, kwargs in cases:  # type: ignore[assignment]
        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )


OP_DB: List[opinfo_core.OpInfo] = [
    opinfo_core.OpInfo(
        "convolution",
        aliases=("convolution",),
        aten_name="convolution",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_convolution,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=common_utils.GRADCHECK_NONDET_TOL,
        skips=(),
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "nn.functional.conv3d",
        aliases=("conv3d",),
        aten_name="conv3d",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_conv3d,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=common_utils.GRADCHECK_NONDET_TOL,
        skips=(),
        supports_out=False,
    ),
]
