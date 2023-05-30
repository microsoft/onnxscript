"""
Test data for aten operators which don't exist in PyTorch file:
pytorch/torch/testing/_internal/common_methods_invocations.py.
"""

import functools
import itertools
from typing import Any, List

import torch
from torch import testing as torch_testing
from torch.testing._internal import (
    common_dtype,
    common_methods_invocations,
    common_utils,
)
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


def sample_inputs_layer_norm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # unused
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # Ordered as input shape, normalized_shape, eps
    cases: tuple[tuple[int], tuple[int], float] = (  # type: ignore[assignment]
        ((1, 2, 3), (1, 2, 3), 0.5),
        ((2, 2, 3), (2, 3), -0.5),
        ((1,), (1,), 1e-5),
        ((1, 2), (2,), 1e-5),
        ((0, 1), (1,), 1e-5),
    )

    for input_shape, normalized_shape, eps in cases:  # type: ignore[misc]
        # Shape of weight and bias should be the same as normalized_shape
        weight = make_arg(normalized_shape)  # type: ignore[has-type]
        bias = make_arg(normalized_shape)  # type: ignore[has-type]
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, weight, bias, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, None, bias, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, weight, None, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, None, None, eps),  # type: ignore[has-type]
        )


class _TestParamsMaxPoolEmptyStrideBase:
    # Adapted from https://github.com/pytorch/pytorch/blob/d6d55f8590eab05d2536756fb4efcfb2d07eb81a/torch/testing/_internal/common_methods_invocations.py#L3203
    def __init__(self):
        self.kwargs = {
            "kernel_size": [3],
            "stride": [()],
            "ceil_mode": [True, False],
            "padding": [0, 1],
            "dilation": [1],
        }

        # fmt: off
        self.shapes = [
            [1, 2, None],  # batch
            [2],  # channels
            [3, 6]  # signal
        ]
        # fmt: on

    def _gen_shape(self):
        for shape in itertools.product(*self.shapes):
            # shape[0] is None indicates missing batch dimension
            if shape[0] is None:
                shape = shape[1:]

            yield shape, torch.contiguous_format
            # only 2d (N, C, H, W) rank 4 tensors support channels_last memory format
            if len(self.shapes) == 4 and len(shape) == 4:
                yield shape, torch.channels_last

    def _gen_kwargs(self):
        keys = self.kwargs.keys()
        for values in itertools.product(*self.kwargs.values()):
            yield dict(zip(keys, values))

    def gen_input_params(self):
        yield from itertools.product(self._gen_shape(), self._gen_kwargs())


class _TestParamsMaxPool1dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3,)]
        self.kwargs["stride"] += [(2,)]
        self.kwargs["padding"] += [(1,)]
        self.kwargs["dilation"] += [(1,)]


class _TestParamsMaxPool2dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3, 2)]
        self.kwargs["stride"] += [(2, 1)]
        self.kwargs["padding"] += [(1, 1)]
        self.kwargs["dilation"] += [(1, 2)]

        self.shapes.append([6])


class _TestParamsMaxPool3dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3, 2, 3)]
        self.kwargs["stride"] += [(2, 1, 2)]
        self.kwargs["dilation"] += [(1, 2, 1)]

        self.shapes.append([6])
        self.shapes.append([5])


def sample_inputs_max_pool_empty_strides(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )

    # FIXME: (RuntimeError: non-empty 3D or 4D (batch mode) tensor expected for input)

    params_generator_type_dict = {
        "max_pool1d": _TestParamsMaxPool1dEmptyStride,
        "max_pool2d": _TestParamsMaxPool2dEmptyStride,
        "max_pool3d": _TestParamsMaxPool3dEmptyStride,
    }

    params_generator = params_generator_type_dict[op_info.name]()
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool1d_with_indices(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool1d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool2d_with_indices(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool2d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool3d_with_indices(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool3d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_native_group_norm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # Ordered as input shape, C,N,HxW, and kwargs for group and eps
    cases = (
        ((1, 6, 3), (6,), (6,), 1, 6, 3, {"group": 2, "eps": 0.5}),
        ((2, 6, 3), (6,), (6,), 2, 6, 3, {"group": 3, "eps": -0.5}),
        ((5, 5, 5), (5,), (5,), 5, 5, 5, {"group": 1, "eps": 1e-5}),
        ((5, 8, 10), (8,), (8,), 5, 8, 10, {"group": 4, "eps": 1e-5}),
    )

    for input_shape, weight, bias, N, C, HxW, kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None

        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(
                weight,
                bias,
                N,
                C,
                HxW,
            ),
            kwargs=kwargs,
        )


def sample_inputs_col2im(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    # input_shape, output_size, kernal, dilation, padding, stride
    cases = (
        (
            (1, 12, 12),
            (4, 5),
            (2, 2),
            {"dilation": (1, 1), "padding": (0, 0), "stride": (1, 1)},
        ),
        (
            (1, 8, 30),
            (4, 5),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (1, 1)},
        ),
        (
            (1, 8, 9),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (0, 0), "stride": (1, 1)},
        ),
        (
            (1, 8, 25),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (1, 1)},
        ),
        (
            (1, 8, 9),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (2, 2)},
        ),
        (
            (1, 9, 4),
            (4, 4),
            (3, 3),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (2, 2)},
        ),
        (
            (1, 18, 16),
            (2, 2),
            (1, 1),
            {"dilation": (2, 2), "padding": (3, 3), "stride": (2, 2)},
        ),
    )

    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    for shape, output_size, kernel_size, kwargs in cases:
        tensor = make_arg(shape)
        yield opinfo_core.SampleInput(tensor, args=(output_size, kernel_size), kwargs=kwargs)


OP_DB: List[opinfo_core.OpInfo] = [
    opinfo_core.OpInfo(
        "col2im",
        op=torch.ops.aten.col2im,
        aten_name="col2im",
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_col2im,
        supports_out=False,
    ),
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
        "layer_norm",
        aliases=("layer_norm",),
        aten_name="layer_norm",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_layer_norm,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=common_utils.GRADCHECK_NONDET_TOL,
        skips=(),
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "native_group_norm",
        op=torch.ops.aten.native_group_norm,
        aten_name="native_group_norm",
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_native_group_norm,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "max_pool1d",
        variant_test_name="empty_strides",
        op=torch.ops.aten.max_pool1d,
        aten_name="max_pool1d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
    ),
    opinfo_core.OpInfo(
        "max_pool2d",
        variant_test_name="empty_strides",
        op=torch.ops.aten.max_pool2d,
        aten_name="max_pool2d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
    ),
    opinfo_core.OpInfo(
        "max_pool3d",
        variant_test_name="empty_strides",
        op=torch.ops.aten.max_pool3d,
        aten_name="max_pool3d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
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
    opinfo_core.OpInfo(
        "nn.functional.max_pool1d_with_indices",
        aten_name="max_pool1d_with_indices",
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        skips=(),
        sample_inputs_func=sample_inputs_max_pool1d_with_indices,
    ),
    opinfo_core.OpInfo(
        "nn.functional.max_pool2d_with_indices",
        aten_name="max_pool2d_with_indices",
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        skips=(),
        sample_inputs_func=sample_inputs_max_pool2d_with_indices,
    ),
    opinfo_core.OpInfo(
        "nn.functional.max_pool3d_with_indices",
        aten_name="max_pool3d_with_indices",
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        skips=(),
        sample_inputs_func=sample_inputs_max_pool3d_with_indices,
    ),
]
