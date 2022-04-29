# SPDX-License-Identifier: Apache-2.0
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64


@script()
def dft_last_axis(x: FLOAT[...], fft_length: INT64[1], onesided=True) -> FLOAT[...]:
    """
    See PR https://github.com/onnx/onnx/pull/3741/.

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
    """

    # Part 1
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    three = op.Constant(value=make_tensor('three', TensorProto.INT64, [1], [3]))
    four = op.Constant(value=make_tensor('four', TensorProto.INT64, [1], [4]))
    last = op.Constant(value=make_tensor('last', TensorProto.INT64, [1], [-1]))
    shape1 = op.Constant(value=make_tensor('shape1', TensorProto.INT64, [2], [-1, 1]))
    shape2 = op.Constant(value=make_tensor('shape2', TensorProto.INT64, [2], [1, -1]))

    nar = op.Range(zero, fft_length, one)  #fft_length or dim
    n0 = op.Cast(nar, to=1)
    n = op.Reshape(n0, shape1)

    kar = op.Range(zero, fft_length, one)
    k0 = op.Cast(kar, to=1)
    k = op.Reshape(k0, shape2)

    cst_2pi = op.Constant(value=make_tensor('pi', TensorProto.FLOAT, [1], [-6.28318530718])) #  -2pi
    fft_length_float = op.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_win = op.Cos(p)
    sin_win = op.Sin(p)

    # Part 2
    if onesided:
        # rfft: x is a float tensor
        x_shape = op.Shape(x)
        axis = op.Size(x_shape) - one
        dim = op.Slice(x_shape, axis, axis + one)

        if dim >= fft_length:
            # fft_length is shorter, x is trimmed to that size
            pad_x = op.Slice(x, zero, fft_length, last, one)
        else:
            if dim == fft_length:
                # no padding
                pad_x = op.Identity(x)
            else:
                # the matrix is completed with zeros
                # operator Pad could be used too.
                x_shape_but_last = op.Slice(op.Shape(x), zero, last, zero, one)
                new_shape = op.Concat(x_shape_but_last, fft_length - dim, axis=0)
                cst = op.ConstantOfShape(new_shape, value=0)
                pad_x = op.Concat(x, op.Cast(cst, to=1), axis=-1)

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
                cst = op.ConstantOfShape(new_shape, value=0)
                pad_r = op.Concat(real_x, op.Cast(cst, to=1), axis=-1)
                pad_i = op.Concat(imag_x, op.Cast(cst, to=1), axis=-1)

        result_real = op.Unsqueeze(op.Sub(op.MatMul(pad_r, cos_win), op.MatMul(pad_i, sin_win)), zero)
        result_imag = op.Unsqueeze(op.Add(op.MatMul(pad_r, sin_win), op.MatMul(pad_i, cos_win)), zero)

    # final step, needs to move to first axis into the last position.
    result = op.Concat(result_real, result_imag, axis=0)
    n_dims = op.Size(op.Shape(result))

    if n_dims == one:
        # This should not happen.
        final = op.Identity(result)
    else:
        result_shape = op.Shape(result)
        shape_cpl = op.Constant(value=make_tensor('shape_cpl', TensorProto.INT64, [2], [2, -1]))
        reshaped_result = op.Reshape(result, shape_cpl)
        transposed = op.Transpose(reshaped_result, perm=[1, 0])
        other_dimensions = op.Slice(result_shape, one, op.Shape(result_shape), zero)
        final_shape = op.Concat(other_dimensions, two, axis=0)
        final = op.Reshape(transposed, final_shape)

    return final


@script()
def switch_axes(x, axis1, axis2):
    """
    Switches two axis. The function assumes `axis1 < axis2`.
    """
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    shape = op.Shape(x)
    n_dims = op.Shape(shape)
    axis2_1 = op.Sub(axis2, one)
    n_dims_1 = op.Sub(n_dims, one)
    
    # First into a 5D dimension tensor.
    dims1_final = op.Slice(shape, zero, axis1, zero)
    if axis1 == zero:
        dims1 = one
    else:
        dims1 = dims1_final

    dims2_final = op.Slice(shape, op.Add(axis1, one), axis2, zero)
    if axis1 == axis2_1:
        dims2 = one
    else:
        dims2 = dims2_final

    dims3_final = op.Slice(shape, op.Add(axis2, one), n_dims, zero)
    if axis2 == n_dims_1:
        dims3 = one
    else:
        dims3 = dims3_final

    dim1 = op.Slice(shape, axis1, op.Add(axis1, one), zero)
    dim2 = op.Slice(shape, axis2, op.Add(axis2, one), zero)

    new_shape = op.Concat(op.ReduceProd(dims1),
                          dim1,
                          op.ReduceProd(dims2),
                          dim2,
                          op.ReduceProd(dims3),
                          axis=0)
    reshaped = op.Reshape(x, new_shape)

    # Transpose
    transposed = op.Transpose(reshaped, perm=[0, 3, 2, 1, 4])

    # Reshape into its final shape.
    final_shape = op.Concat(dims1_final, dim2, dims2_final, dim1, dims3_final, axis=0)
    return op.Reshape(transposed, final_shape)


@script()
def dft(x: FLOAT[...], fft_length: INT64[1], axis: INT64[1], onesided=True) -> FLOAT[...]:
    """
    Applies one dimension FFT.
    The function moves the considered axis to the last position
    calls dft_last_axis, and moves the axis to its original position.
    """
    shape = op.Shape(x)
    n_dims = op.Shape(shape)

    if onesided:
        one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
        last_dim = op.Sub(n_dims, one)
    else:
        two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
        last_dim = op.Sub(n_dims, two)

    if axis == last_dim:
        final = dft_last_axis(x, fft_length, onesided)
    else:
        xt = switch_axes(x, axis, last_dim)
        fft = dft_last_axis(xt, fft_length, onesided)
        final = switch_axes(fft, axis, last_dim)
    return final


# if __name__ == "__main__":
#     import numpy as np
#     x = np.array([[0, 1, 2, 3, 4],
#                   [5, 6, 7, 8, 9]],
#                  dtype=np.float32)
#     ax1 = np.array([0], dtype=np.int64)
#     le = np.array([4], dtype=np.int64)
#     r = dft(x, le, ax1, True)
#     print(r.shape)
#     print(r)
