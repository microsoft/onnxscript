# SPDX-License-Identifier: Apache-2.0
from onnxscript.onnx_types import FLOAT, INT64


def dft(N: INT64[1], fft_length: INT64[1]) -> FLOAT["I", "J"]:
    # Documentation is not supported yet.
    # """
    # Returns the matrix
    # :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`.
    # """
    one = oxs.Constant(value_float=1)
    zeroi = oxs.Constant(value_int64=0)
    onei = oxs.Constant(value_int64=1)
    minusi = oxs.Neg(onei)  # oxs.Constant(value_int64=-1) fails
    shape1 = oxs.Concat(minusi, onei)  # oxs.Constant(value_floats=[-1, 1])  fails
    shape2 = oxs.Concat(onei, minusi)

    nar = oxs.Range(zeroi, N, one)  
    n0 = oxs.Cast(nar, to=1)
    n1 = oxs.Reshape(n0, to=1)
    n = oxs.Reshape(n1, shape1)
    
    # twice the same code N is replaced by fft_length in the code below
    # how to put in a function.
    kar = oxs.Range(zeroi, fft_length, one)
    k0 = oxs.Cast(kar, to=1)
    k1 = oxs.Reshape(k0, to=1)
    k = oxs.Reshape(k1, shape1)
    
    cst_2pi = oxs.Neg(oxs.Constant(value_float=6.28318530718)) #  -2pi
    fft_length_float = oxs.Cast(fft_length, to=1)
    p = (k / fft_length_float * cst_2pi) * n
    cos_p = oxs.Cos(p)
    sin_p = oxs.Sin(p)
    two = oxs.Constant(value_int64=2)
    new_shape = oxs.Concat(two, oxs.Shape(cos_p))  # unsupported
    cplx = oxs.Concat(cos_p, sin_p)
    return oxs.Reshape(cplx, new_shape)


def dynamic_switch_with_last_axis(x: FLOAT[None], axis: INT64[1]) -> FLOAT[None]:
    # transpose with the permutation as an attribute does not
    # work here, we need a permutation depending on the input data
    zero = oxs.Constant(value_int64=0)
    one = oxs.Constant(value_int64=1)
    dim = oxs.Size(oxs.Shape(x)) - oxs.Constant(value_int64=1)  # x.shape.size - 1 or len(x.shape) - 1
    if axis == dim or dim == zero:  # ValueError: Unsupported expression type: BoolOp. (difficult to understand)
        return x
    else:
        if dim == one:
            if axis == zero:
                return oxs.Transpose(x, perm=[1, 0])
            else:  # can we skip else?
                return x  # it is covered by the first case
        else:
            two = oxs.Constant(value_int64=2)
            if dim == two:
                if axis == zero:
                    return oxs.Transpose(x, perm=[2, 1, 0])
                else:
                    return x
                if axis == one:
                    return oxs.Transpose(x, perm=[0, 2, 1])
                else:
                    return x
            else:
                three = oxs.Constant(value_int64=3)
                if dim == three:
                    if axis == zero:
                        return oxs.Transpose(x, perm=[2, 1, 0])
                    else:
                        return x
                    if axis == one:
                        return oxs.Transpose(x, perm=[0, 2, 1])
                    else:
                        return x


def fft(x: FLOAT[None], fft_length: INT64, axis: INT64) -> FLOAT[None]:
    # Similar to numpy.fft
    # one dimension.
    cst = dft(new_x.shape[axis], length)
    cst_cast = oxs.CastLike(cst, x)
    xt = dynamic_switch_with_last_axis(x, axis)

    step = oxs.Constant(value_int64=1)
    dims = oxs.Shape(xt)
    xt_shape = xt.shape
    if xt_shape[axis] >= fft_length:
        new_xt = oxs.Slice(xt, oxs.Constant(value_int64=0), dims[-1], axis, step)
    elif xt_shape[axis] == fft_length:
        new_xt = xt
    else:
        # other, the matrix is completed with zeros
        delta = length - xt_shape[-1]
        new_shape = oxs.Concat(xt_shape[:-1], delta)
        zero = oxs.ConstantOfShape(new_shape, value=0)
        new_xt = concat(xt, zero)

    result = oxs.MatMul(xt, cst_cast)
    xt = dynamic_switch_with_last_axis(x, axis)
    return xt
