# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64

PI = np.pi
TWO_PI = np.pi * 2
FOUR_PI = np.pi * 4


@script()
def hann_window(window_length):
    """Returns
    :math:`\\omega_n = \\sin^2\\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.
    """
    N_minus_1 = op.Cast(window_length - 1, to=1)
    ni = op.Cast(op.Range(0, window_length, 1), to=1)
    pin = (ni * PI) / N_minus_1
    sin = op.Sin(pin)
    return sin * sin


@script()
def hamming_window(window_length, alpha, beta):
    """Returns
    :math:`\\omega_n = \\alpha - \\beta \\cos \\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.

    Default values for torch: `alpha=0.54, beta=0.46`.
    """
    N_minus_1 = op.Cast(window_length - 1, to=1)
    ni = op.Cast(op.Range(0, window_length, 1), to=1)
    pin = (ni * TWO_PI) / N_minus_1
    cos = op.Cos(pin)
    return alpha - cos * beta


@script()
def blackman_window(window_length):
    """Returns
    :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) +
    0.8 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.
    """
    N_minus_1 = op.Cast(window_length - 1, to=1)
    ni = op.Cast(op.Range(0, window_length, 1), to=1)
    cos2 = op.Cos((ni * TWO_PI) / N_minus_1)
    cos4 = op.Cos((ni * FOUR_PI) / N_minus_1)
    return (0.42 - (cos2 * 0.5)) + (cos4 * 0.08)


@script()
def switch_axes(x: FLOAT[...], axis1: INT64[1], axis2: INT64[1]) -> FLOAT[...]:
    """Switches two axis. The function assumes `axis1 < axis2`.
    Both axis1 and axis2 are assumed to be positive. Specifically, the convention
    of using negative axes to count backwards from the end is not supported.
    """
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor("one", TensorProto.INT64, [1], [1]))
    shape = op.Shape(x)
    n_dims = op.Shape(shape)
    axis2_1 = axis2 - one
    n_dims_1 = n_dims - one

    # First into a 5D dimension tensor.
    pre_axis1 = op.Slice(shape, zero, axis1, zero)
    if axis1 == zero:
        pre_axis1_size = op.Identity(one)
    else:
        pre_axis1_size = op.ReduceProd(pre_axis1)

    between = op.Slice(shape, op.Add(axis1, one), axis2, zero)
    if axis1 == axis2_1:
        between_size = op.Identity(one)
    else:
        between_size = op.ReduceProd(between)

    post_axis2 = op.Slice(shape, op.Add(axis2, one), n_dims, zero)
    if axis2 == n_dims_1:
        post_axis2_size = op.Identity(one)
    else:
        post_axis2_size = op.ReduceProd(post_axis2)

    dim1_size = op.Slice(shape, axis1, op.Add(axis1, one), zero)
    dim2 = op.Slice(shape, axis2, op.Add(axis2, one), zero)

    new_shape = op.Concat(
        pre_axis1_size,
        dim1_size,
        between_size,
        dim2,
        post_axis2_size,
        axis=0,
    )
    reshaped = op.Reshape(x, new_shape)

    # Transpose
    transposed = op.Transpose(reshaped, perm=[0, 3, 2, 1, 4])

    # Reshape into its final shape.
    final_shape = op.Concat(pre_axis1, dim2, between, dim1_size, post_axis2, axis=0)
    return op.Reshape(transposed, final_shape)


@script()
def dft_last_axis(
    x: FLOAT[...],
    fft_length: INT64[1],
    onesided: bool = False,
    inverse: bool = False,
    normalize: bool = False,
) -> FLOAT[...]:
    """See PR https://github.com/onnx/onnx/pull/3741/.

    *Part 1*

    Computes the matrix:
    :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`
    and builds two matrices, real part and imaginary part.

    *Part 2*

    Matrix multiplication. The fft axis is the last one.
    It builds two matrices, real and imaginary parts for DFT.

    *Part 3*

    Part 2 merges the real and imaginary parts into one single matrix
    where the last axis indicates whether it is the real or the imaginary part.

    Args:
        x: float tensor, the last dimension is the complex one, it has 1
            or 2 elements, 1 if the tensor is real and does not have any
            imaginary part, 2 if the tensor is complex
        fft_length: length of the FFT
        onesided: if True, returns a truncated result `[:fft_length//2]`
        inverse: returns FFT or the inverse of FFT
        normalize: normalizes the result

    Returns:
        tensor
    """

    # Part 1
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor("one", TensorProto.INT64, [1], [1]))
    two = op.Constant(value=make_tensor("two", TensorProto.INT64, [1], [2]))
    last = op.Constant(value=make_tensor("last", TensorProto.INT64, [1], [-1]))

    range = op.Range(zero, fft_length, one)  # fft_length or dim
    range_float = op.Cast(range, to=1)
    shape1 = op.Constant(value=make_tensor("shape1", TensorProto.INT64, [2], [-1, 1]))
    n = op.Reshape(range_float, shape1)

    shape2 = op.Constant(value=make_tensor("shape2", TensorProto.INT64, [2], [1, -1]))
    k = op.Reshape(range_float, shape2)

    if op.Cast(inverse, to=TensorProto.BOOL):
        cst_2pi = op.Constant(
            value=make_tensor("pi", TensorProto.FLOAT, [1], [math.tau])
        )  #  2pi
    else:
        cst_2pi = op.Constant(
            value=make_tensor("pi", TensorProto.FLOAT, [1], [-math.tau])
        )  #  -2pi
    fft_length_float = op.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_win = op.Cos(p)
    sin_win = op.Sin(p)

    # real or complex
    last_dim = op.Shape(x, start=-1)

    # Part 2
    if last_dim == one:
        # rfft: x is a float tensor
        real_x = op.Squeeze(op.Slice(x, zero, one, last), last)
        x_shape = op.Shape(real_x)
        axis = op.Size(x_shape) - one
        dim = op.Slice(x_shape, axis, axis + one)

        if dim >= fft_length:
            # fft_length is shorter, x is trimmed to that size
            pad_x = op.Slice(real_x, zero, fft_length, last, one)
        else:
            if dim == fft_length:
                # no padding
                pad_x = op.Identity(real_x)
            else:
                # the matrix is completed with zeros
                # operator Pad could be used too.
                x_shape_but_last = op.Slice(op.Shape(real_x), zero, last, zero, one)
                new_shape = op.Concat(x_shape_but_last, fft_length - dim, axis=0)
                cst = op.ConstantOfShape(
                    new_shape, value=make_tensor("zerof", TensorProto.FLOAT, [1], [0])
                )
                pad_x = op.Concat(real_x, op.Cast(cst, to=1), axis=-1)

        result_real = op.Unsqueeze(op.MatMul(pad_x, cos_win), zero)
        result_imag = op.Unsqueeze(op.MatMul(pad_x, sin_win), zero)

    else:
        # fft: x is a complex tensor in a float tensor
        # last dimension is the complex one
        x_shape_c = op.Shape(x)
        x_shape = op.Slice(x_shape_c, zero, last, last)
        axis = op.Size(x_shape) - one
        dim = op.Slice(x_shape, axis, axis + one)

        real_x = op.Squeeze(op.Slice(x, zero, one, last), last)
        imag_x = op.Squeeze(op.Slice(x, one, two, last), last)

        if dim >= fft_length:
            # fft_length is shorter, x is trimmed to that size
            pad_r = op.Slice(real_x, zero, fft_length, last, one)
            pad_i = op.Slice(imag_x, zero, fft_length, last, one)
        else:
            if dim == fft_length:
                # no padding
                pad_r = op.Identity(real_x)
                pad_i = op.Identity(imag_x)
            else:
                # the matrix is completed with zeros
                # operator Pad could be used too.
                x_shape_but_last = op.Slice(op.Shape(real_x), zero, last, zero, one)
                new_shape = op.Concat(x_shape_but_last, fft_length - dim, axis=0)
                cst = op.ConstantOfShape(
                    new_shape, value=make_tensor("zerof", TensorProto.FLOAT, [1], [0])
                )
                pad_r = op.Concat(real_x, op.Cast(cst, to=1), axis=-1)
                pad_i = op.Concat(imag_x, op.Cast(cst, to=1), axis=-1)

        result_real = op.Unsqueeze(
            op.Sub(op.MatMul(pad_r, cos_win), op.MatMul(pad_i, sin_win)), zero
        )
        result_imag = op.Unsqueeze(
            op.Add(op.MatMul(pad_r, sin_win), op.MatMul(pad_i, cos_win)), zero
        )

    # final step, needs to move to first axis into the last position.
    result = op.Concat(result_real, result_imag, axis=0)
    n_dims = op.Size(op.Shape(result))

    if op.Cast(onesided, to=TensorProto.BOOL):
        half = op.Div(fft_length, two) + op.Mod(fft_length, two)
        n_r_dims_1 = op.Sub(op.Shape(op.Shape(x)), one)
        truncated = op.Slice(result, zero, half, n_r_dims_1)
    else:
        truncated = op.Identity(result)

    if n_dims == one:
        # This should not happen.
        final = op.Identity(truncated)
    else:
        result_shape = op.Shape(truncated)
        shape_cpl = op.Constant(
            value=make_tensor("shape_cpl", TensorProto.INT64, [2], [2, -1])
        )
        reshaped_result = op.Reshape(truncated, shape_cpl)
        transposed = op.Transpose(reshaped_result, perm=[1, 0])
        other_dimensions = op.Slice(result_shape, one, op.Shape(result_shape), zero)
        final_shape = op.Concat(other_dimensions, two, axis=0)
        final = op.Reshape(transposed, final_shape)

    # normalization is needed for idft.
    if op.Cast(normalize, to=TensorProto.BOOL):
        norm = op.Div(final, fft_length_float)
    else:
        norm = op.Identity(final)
    return norm


@script()
def dft_inv(
    x: FLOAT[...],
    fft_length: INT64[1],
    axis: INT64[1],
    onesided: bool = False,
    inverse: bool = False,
    normalize: bool = False,
) -> FLOAT[...]:
    """Applies one dimension FFT.

    The function moves the considered axis to the last position
    calls dft_last_axis, and moves the axis to its original position.
    """
    shape = op.Shape(x)
    n_dims = op.Shape(shape)
    last_dim = n_dims - 2
    positive_axis = op.Where(axis < 0, axis + n_dims, axis)

    if positive_axis == last_dim:
        final = dft_last_axis(x, fft_length, onesided, inverse, normalize)
    else:
        xt = switch_axes(x, positive_axis, last_dim)
        fft = dft_last_axis(xt, fft_length, onesided, inverse, normalize)
        final = switch_axes(fft, positive_axis, last_dim)
    return final


@script(default_opset=op)
def dft(
    x: FLOAT[...],
    fft_length: INT64[1],
    axis: INT64[1],
    inverse: bool = False,
    onesided: bool = False,
) -> FLOAT[...]:
    """Applies one dimensional FFT.

    The function moves the considered axis to the last position
    calls dft_last_axis, and moves the axis to its original position.
    """
    return dft_inv(x, fft_length, axis, onesided=onesided, inverse=inverse, normalize=inverse)


@script()
def stft(
    x: FLOAT[...],
    fft_length: INT64[1],
    hop_length: INT64[1],
    n_frames: INT64[1],
    window: FLOAT["N"],
    onesided: bool = False,
) -> FLOAT[...]:
    """Applies one dimensional FFT with window weights.

    torch defines the number of frames as:
    `n_frames = 1 + (len - n_fft) / hop_length`.
    """
    one = op.Constant(value=make_tensor("one", TensorProto.INT64, [1], [1]))
    mtwo = op.Constant(value=make_tensor("mtwo", TensorProto.INT64, [1], [-2]))
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    last_axis = op.Shape(op.Shape(x)) - one
    axis = op.Constant(value=make_tensor("axis", TensorProto.INT64, [1], [-2]))
    axis2 = op.Constant(value=make_tensor("axis2", TensorProto.INT64, [1], [-3]))
    window_size = op.Shape(window)

    # building frames
    seq = op.SequenceEmpty(dtype=TensorProto.FLOAT)
    nf = op.Squeeze(n_frames, zero)
    for fs in range(nf):
        fs64 = op.Cast(fs, to=7)
        begin = fs64 * hop_length
        end = begin + window_size
        sliced_x = op.Slice(x, begin, end, axis)

        # sliced_x may be smaller
        new_dim = op.Shape(sliced_x, start=-2, end=-1)
        missing = window_size - new_dim
        new_shape = op.Concat(
            op.Shape(sliced_x, start=0, end=-2),
            missing,
            op.Shape(sliced_x, start=-1),
            axis=0,
        )
        cst = op.ConstantOfShape(
            new_shape, value=make_tensor("zerof", TensorProto.FLOAT, [1], [0])
        )
        pad_sliced_x = op.Concat(sliced_x, op.Cast(cst, to=1), axis=-2)

        # same size
        un_sliced_x = op.Unsqueeze(pad_sliced_x, axis2)
        seq = op.SequenceInsert(seq, un_sliced_x)

    # concatenation
    new_x = op.ConcatFromSequence(seq, axis=-3, new_axis=0)

    # calling weighted dft with weights=window
    shape_x = op.Shape(new_x)
    shape_x_short = op.Slice(shape_x, zero, mtwo, zero)
    shape_x_short_one = (shape_x_short * zero) + one
    window_shape = op.Concat(shape_x_short_one, window_size, one, axis=0)
    weights = op.Reshape(window, window_shape)
    weighted_new_x = new_x * weights

    result = dft(weighted_new_x, fft_length, last_axis, onesided, False)

    # final transpose -3, -2
    two = op.Constant(value=make_tensor("two", TensorProto.INT64, [1], [2]))
    three = op.Constant(value=make_tensor("three", TensorProto.INT64, [1], [3]))
    dim = op.Shape(op.Shape(result))
    return switch_axes(result, dim - three, dim - two)
