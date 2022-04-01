# SPDX-License-Identifier: Apache-2.0
# docstring is not support here.
import numpy as np
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript import eager_mode_evaluator as oxs


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

    one = oxs.Constant(value_floats=[1.])
    zeroi = oxs.Constant(value_ints=[0])
    onei = oxs.Constant(value_ints=[1])
    minusi = oxs.Neg(onei)  # oxs.Constant(value_int64=-1) fails
    print([minusi, onei])
    shape1 = oxs.Concat(minusi, onei, axis=0)  # oxs.Constant(value_floats=[-1, 1])  fails
    shape2 = oxs.Concat(onei, minusi, axis=0)

    nar = oxs.Range(zeroi, N, one)
    n0 = oxs.Cast(nar, to=1)
    n = oxs.Reshape(n0, shape1)
    
    kar = oxs.Range(zeroi, fft_length, one)
    k0 = oxs.Cast(kar, to=1)
    k = oxs.Reshape(k0, shape2)
    
    cst_2pi = oxs.Neg(oxs.Constant(value_floats=[6.28318530718])) #  -2pi
    fft_length_float = oxs.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_p = oxs.Cos(p)
    sin_p = oxs.Sin(p)
    two = oxs.Constant(value_ints=[2])
    new_shape = oxs.Concat(two, oxs.Shape(cos_p), axis=0)  # unsupported
    cplx = oxs.Concat(cos_p, sin_p, axis=0)
    return oxs.Reshape(cplx, new_shape)


n = np.array([3], dtype=np.int64)
print(dft(n, n))


def dynamic_switch_with_last_axis(x: FLOAT[None], axis: INT64[1]) -> FLOAT[None]:
    # transpose with the permutation as an attribute does not
    # work here, we need a permutation depending on the input data
    zero = oxs.Constant(value_ints=[0])
    one = oxs.Constant(value_ints=[1])
    two = oxs.Constant(value_ints=[2])
    three = oxs.Constant(value_ints=[3])
    dim = oxs.Size(oxs.Shape(x)) - oxs.Constant(value_ints=[1])  # x.shape.size - 1 or len(x.shape) - 1
    if axis == dim or dim == zero:
        result = x
    else:
        if dim == one:  # Error: Variable result is not assigned a value along a conditional branch
            if axis == zero:
                result = oxs.Transpose(x, perm=[1, 0])
            else:  # can we skip else?
                result = x  # it is covered by the first case
        else:
            if dim == two:
                if axis == zero:
                    result = oxs.Transpose(x, perm=[2, 1, 0])
                else:
                    result = x
                if axis == one:
                    result = oxs.Transpose(x, perm=[0, 2, 1])
                else:
                    result = x
            else:
                # three = oxs.Constant(value_int64=3)  # cannot declare local variables
                if dim == three:
                    if axis == zero:
                        result = oxs.Transpose(x, perm=[2, 1, 0])
                    else:
                        result = x
                    if axis == one:
                        result = oxs.Transpose(x, perm=[0, 2, 1])
                    else:
                        result = x
                else:
                    result = x
    return result


def fft(x: FLOAT[None], fft_length: INT64, axis: INT64) -> FLOAT[None]:
    # Similar to numpy.fft
    # one dimension.
    # Simpler to write with [axis].
    # cst = dft(x.shape[axis], length)  # dft is unknown, subfunction are not allowed
    x_shape = oxs.Shape(x)
    dim = oxs.Slice(x_shape, axis, axis + oxs.Constant(value_ints=[1]))
    cst = dft(dim, fft_length)
    cst_cast = oxs.CastLike(cst, x)
    step = oxs.Constant(value_ints=[1])
    last = oxs.Neg(oxs.Constant(value_ints=[1]))  # value_int64=-1 calls UnaryOp
    zero_i = oxs.Constant(value_ints=[0])
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
