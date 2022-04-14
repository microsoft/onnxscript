# SPDX-License-Identifier: Apache-2.0
# docstring is not support here.
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx import opset15 as op


# infer_shapes: RuntimeError: Input 1 is out of bounds. (no clue about what is wrong)
def dft(N: INT64[1], fft_length: INT64[1]) -> FLOAT["I", "J"]:
    # Documentation is not supported yet.
    # """
    # Returns the matrix
    # :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`.
    # """
    # numpy code
    # 
    # def dft(N, fft_length):
    #     """
    #     Returns the matrix
    #     :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`.
    #     """
    #     dtype = numpy.float64
    #     zero = numpy.array([0], dtype=numpy.int64)
    #     n = arange(zero, N).astype(dtype).reshape((-1, 1))
    #     k = arange(zero, fft_length).astype(dtype).reshape((1, -1))
    #     p = (k / fft_length.astype(dtype=dtype) *
    #          numpy.array([-numpy.pi * 2], dtype=dtype)) * n
    #     cos_p = cos(p)
    #     sin_p = sin(p)
    #     two = numpy.array([2], dtype=numpy.int64)
    #     new_shape = concat(two, cos_p.shape)
    #     return concat(cos_p, sin_p).reshape(new_shape)

    zeroi = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    shape1 = op.Constant(value=make_tensor('shape1', TensorProto.INT64, [2], [-1, 1]))
    shape2 = op.Constant(value=make_tensor('shape2', TensorProto.INT64, [2], [1, -1]))

    nar = op.Range(zeroi, N, one)
    n0 = op.Cast(nar, to=1)
    n = op.Reshape(n0, shape1)
    
    kar = op.Range(zeroi, fft_length, one)
    k0 = op.Cast(kar, to=1)
    k = op.Reshape(k0, shape2)
    
    cst_2pi = op.Constant(value=make_tensor('pi', TensorProto.FLOAT, [1], [-6.28318530718])) #  -2pi
    fft_length_float = op.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_p = op.Cos(p)
    sin_p = op.Sin(p)
    shape = op.Shape(cos_p)
    new_shape = op.Concat(two, shape, axis=0)  # unsupported
    cplx = op.Concat(cos_p, sin_p, axis=0)
    return op.Reshape(cplx, new_shape)


def dynamic_switch_with_last_axis_2d(x: FLOAT[None, None], axis: INT64[1]) -> FLOAT[None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    if axis == zero:
        result = op.Transpose(x, perm=[1, 0])
    else:  # can we skip else?
        result = op.Identity(x)
    return result


def dynamic_switch_with_last_axis_3d(x: FLOAT[None, None, None], axis: INT64[1]) -> FLOAT[None, None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    if axis == zero:
        result = op.Transpose(x, perm=[2, 1, 0])
    else:
        if axis == one:
            result = op.Transpose(x, perm=[0, 2, 1])
        else:
            result = op.Identity(x)
    return result


def dynamic_switch_with_last_axis_4d(x: FLOAT[None, None, None, None], axis: INT64[1]) -> FLOAT[None, None, None, None]:
    zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    if axis == zero:
        result = op.Transpose(x, perm=[3, 1, 2, 0])
    else:
        if axis == one:
            result = op.Transpose(x, perm=[0, 3, 2, 1])
        else:
            if axis == two:
                result = op.Transpose(x, perm=[0, 1, 3, 2])
            else:
                result = op.Identity(x)
    return result


# def dynamic_switch_with_last_axis(x: FLOAT[...], axis: INT64[1]) -> FLOAT[...]:
#     # transpose with the permutation as an attribute does not
#     # work here, we need a permutation depending on the input data
#     zero = op.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
#     one = op.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
#     two = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
#     three = op.Constant(value=make_tensor('three', TensorProto.INT64, [1], [3]))
#     dim = op.Size(op.Shape(x)) - one  # x.shape.size - 1 or len(x.shape) - 1
#     if not(axis == dim) and dim > zero:
#         if dim == one:  # Error: Variable result is not assigned a value along a conditional branch
#             result = dynamic_switch_with_last_axis_2d(x, axis)
#         else:
#             if dim == two:
#                 result = dynamic_switch_with_last_axis_3d(x, axis)
#             else:
#                 result = dynamic_switch_with_last_axis_4d(x, axis)
#     else:
#         result = op.Identity(x)
#     return result


def fft(x: FLOAT[None], fft_length: INT64, axis: INT64) -> FLOAT[None]:
    # Similar to numpy.fft
    # one dimension.
    # Simpler to write with [axis].
    # cst = dft(x.shape[axis], length)  # dft is unknown, subfunction are not allowed
    step = op.Constant(value=make_tensor('step', TensorProto.INT64, [1], [1]))
    last = op.Constant(value=make_tensor('last', TensorProto.INT64, [1], [-1]))
    zero_i = op.Constant(value=make_tensor('last', TensorProto.INT64, [1], [0]))

    x_shape = op.Shape(x)
    dim = op.Slice(x_shape, axis, axis + step)
    cst = dft(dim, fft_length)
    cst_cast = op.CastLike(cst, x)
    # xt = dynamic_switch_with_last_axis(x, axis)
    xt = dynamic_switch_with_last_axis_4d(x, axis)
    
    # Cannot create variable inside a branch of a test
    xt_shape_but_last = op.Slice(op.Shape(xt), zero_i, last, zero_i, step)
    new_shape = op.Concat(xt_shape_but_last, fft_length - dim, axis=0)

    if dim >= fft_length:
        print("A", dim, fft_length, zero_i, fft_length, last, step)
        print("xt", xt.shape)
        new_xt = op.Slice(xt, zero_i, fft_length, last, step)
    else:
        print("B")
        if dim == fft_length:  # not sure about elif
            print("C")
            new_xt = op.Identity(xt)
        else:
            # other, the matrix is completed with zeros
            print("D")
            new_xt = op.Concat(xt, op.ConstantOfShape(new_shape, value=0))

    print("D", new_xt.shape, cst_cast.shape)
    result = op.MatMul(new_xt, cst_cast)
    # final = dynamic_switch_with_last_axis(xt, axis)
    final = dynamic_switch_with_last_axis_4d(result, axis)
    return final


if __name__ == "__main__":
    import numpy as np
    from numpy.testing import assert_almost_equal

    x = np.random.randn(4, 4, 4, 4).astype(np.float32)
    l = np.array([4], dtype=np.int64)
    a = np.array([3], dtype=np.int64)
    result = fft(x, l, a)
    print(result)
    print('done')
    