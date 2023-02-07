"""
Test data for aten operators which don't exist in PyTorch file:
pytorch/torch/testing/_internal/common_methods_invocations.py.
"""

import functools
from typing import Tuple, List

import torch
from torch.testing import make_tensor

from torch.testing._internal.common_cuda import CUDA11OrLater
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM, GRADCHECK_NONDET_TOL
)
from torch.testing._internal.opinfo import core as opinfo_core


def sample_inputs_convolution(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: Tuple = (
        ((1, 3, 4), (3, 3, 3), (3,),
            {'stride': (2,), 'padding': (2,), 'dilation': (1,), 'transposed': False, 'output_padding': (0,), 'groups': 1}),
        ((1, 3, 4), (3, 3, 3), None,
            {'stride': (2,), 'padding': (2,), 'dilation': (1,), 'transposed': True, 'output_padding': (0,), 'groups': 1}),
        ((1, 3, 224, 224), (32, 3, 3, 3), None,
            {'stride': (2, 2), 'padding': (1, 1), 'dilation': (1, 1), 'transposed': False, 'output_padding': (0, 0), 'groups': 1}),
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,),
            {'stride': (3, 2), 'padding': (1, 1), 'dilation': (1, 1), 'transposed': True, 'output_padding': (0, 0), 'groups': 4}),
    )

    for input_shape, weight, bias, kwargs in cases:
        yield opinfo_core.SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)


op_db: List[opinfo_core.OpInfo] = [
    opinfo_core.OpInfo('convolution',
           aliases=('convolution',),
           aten_name='convolution',
           dtypes=floating_and_complex_types_and(torch.int64, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf,
                                                       *[torch.bfloat16] if (CUDA11OrLater or TEST_WITH_ROCM) else []),
           sample_inputs_func=sample_inputs_convolution,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_jit_shape_analysis=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(),
           supports_expanded_weight=True,
           supports_out=False,),
]
