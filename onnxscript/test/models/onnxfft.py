# SPDX-License-Identifier: Apache-2.0
# docstring is not support here.
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript.onnx_types import FLOAT, INT64


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

    zeroi = oxs.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = oxs.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = oxs.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    shape1 = oxs.Constant(value=make_tensor('shape1', TensorProto.INT64, [2], [-1, 1]))
    shape2 = oxs.Constant(value=make_tensor('shape2', TensorProto.INT64, [2], [1, -1]))

    nar = oxs.Range(zeroi, N, one)
    n0 = oxs.Cast(nar, to=1)
    n = oxs.Reshape(n0, shape1)
    
    kar = oxs.Range(zeroi, fft_length, one)
    k0 = oxs.Cast(kar, to=1)
    k = oxs.Reshape(k0, shape2)
    
    cst_2pi = oxs.Constant(value=make_tensor('pi', TensorProto.FLOAT, [1], [-6.28318530718])) #  -2pi
    fft_length_float = oxs.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_p = oxs.Cos(p)
    sin_p = oxs.Sin(p)
    shape = oxs.Shape(cos_p)
    new_shape = oxs.Concat(two, shape, axis=0)  # unsupported
    cplx = oxs.Concat(cos_p, sin_p, axis=0)
    return oxs.Reshape(cplx, new_shape)


def dynamic_switch_with_last_axis(x: FLOAT[None], axis: INT64[1]) -> FLOAT[None]:
    # transpose with the permutation as an attribute does not
    # work here, we need a permutation depending on the input data
    zero = oxs.Constant(value=make_tensor('zero', TensorProto.INT64, [1], [0]))
    one = oxs.Constant(value=make_tensor('one', TensorProto.INT64, [1], [1]))
    two = oxs.Constant(value=make_tensor('two', TensorProto.INT64, [1], [2]))
    three = oxs.Constant(value=make_tensor('three', TensorProto.INT64, [1], [3]))
    dim = oxs.Size(oxs.Shape(x)) - one  # x.shape.size - 1 or len(x.shape) - 1
    if axis == dim or dim == zero:
        result = oxs.Identity(x)  # result = x does not work yet
    else:
        if dim == one:  # Error: Variable result is not assigned a value along a conditional branch
            if axis == zero:
                result = oxs.Transpose(x, perm=[1, 0])
            else:  # can we skip else?
                result = oxs.Identity(x)  # result = x does not work yet
        else:
            if dim == two:
                if axis == zero:
                    result = oxs.Transpose(x, perm=[2, 1, 0])
                else:
                    result = oxs.Identity(x)  # result = x does not work yet
                if axis == one:
                    result = oxs.Transpose(x, perm=[0, 2, 1])
                else:
                    result = oxs.Identity(x)  # result = x does not work yet
            else:
                # three = oxs.Constant(value_int64=3)  # cannot declare local variables
                if dim == three:
                    if axis == zero:
                        result = oxs.Transpose(x, perm=[2, 1, 0])
                    else:
                        result = oxs.Identity(x)  # result = x does not work yet
                    if axis == one:
                        result = oxs.Transpose(x, perm=[0, 2, 1])
                    else:
                        result = oxs.Identity(x)  # result = x does not work yet
                else:
                    result = oxs.Identity(x)  # result = x does not work yet
    return result


def fft(x: FLOAT[None], fft_length: INT64, axis: INT64) -> FLOAT[None]:
    # Similar to numpy.fft
    # one dimension.
    # Simpler to write with [axis].
    # cst = dft(x.shape[axis], length)  # dft is unknown, subfunction are not allowed
    step = oxs.Constant(value=make_tensor('step', TensorProto.INT64, [1], [1]))
    last = oxs.Constant(value=make_tensor('last', TensorProto.INT64, [1], [-1]))
    zero_i = oxs.Constant(value=make_tensor('last', TensorProto.INT64, [1], [0]))

    x_shape = oxs.Shape(x)
    dim = oxs.Slice(x_shape, axis, axis + step)
    cst = dft(dim, fft_length)
    cst_cast = oxs.CastLike(cst, x)
    xt = dynamic_switch_with_last_axis(x, axis)

    # Cannot create variable inside a branch of a test
    xt_shape_but_last = oxs.Slice(oxs.Shape(xt), zero_i, last, zero_i, step)
    new_shape = oxs.Concat(xt_shape_but_last, fft_length - dim)

    if dim >= fft_length:
        new_xt = oxs.Slice(xt, zero_i, fft_length, last, step)
    else:
        if dim == fft_length:  # not sure about elif
            new_xt = xt
        else:
            # other, the matrix is completed with zeros
            new_xt = oxs.Concat(xt, oxs.ConstantOfShape(new_shape, value=0))

    result = oxs.MatMul(xt, cst_cast)
    final = dynamic_switch_with_last_axis(xt, axis)
    return final


if __name__ == "__main__":
    import numpy as np
    from numpy.testing import assert_almost_equal
    from onnxscript import eager_mode_evaluator as oxs

    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    result = dynamic_switch_with_last_axis(x, np.array([0], dtype=np.int64))
    expected = x.T
    assert_almost_equal(expected, result)
    print('done')
    