# SPDX-License-Identifier: Apache-2.0
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.onnx import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64


@script()
def dft(x: FLOAT[...], fft_length: INT64[1], onesided=True) -> FLOAT[...]:
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
    three = op.Constant(value=make_tensor('two', TensorProto.INT64, [1], [3]))
    last = op.Constant(value=make_tensor('last', TensorProto.INT64, [1], [-1]))
    shape1 = op.Constant(value=make_tensor('shape1', TensorProto.INT64, [2], [-1, 1]))
    shape2 = op.Constant(value=make_tensor('shape2', TensorProto.INT64, [2], [1, -1]))
    x_shape = op.Shape(x)
    axis = op.Size(x_shape) - one
    dim = op.Slice(x_shape, axis, axis + one)

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
        
        if dim >= fft_length:
            pad_x = op.Slice(x, zero, fft_length, last, one)
        else:
            if dim == fft_length:  # not sure about elif
                pad_x = op.Identity(x)
            else:
                # other, the matrix is completed with zeros
                x_shape_but_last = op.Slice(op.Shape(x), zero, last, zero, one)
                new_shape = op.Concat(x_shape_but_last, fft_length - dim, axis=0)
                print(x.shape, new_shape)
                
                # The current parser does not support
                # unary operator and `-1` is interpreted as `-(1)`.
                # It produces the error `ValueError: Unsupported attribute type: UnaryOp`.
                pad_x = op.Concat(x, op.ConstantOfShape(new_shape, value=0), axis=-1)

        result_real = op.Unsqueeze(op.MatMul(pad_x, cos_win), zero)
        result_imag = op.Unsqueeze(op.MatMul(pad_x, sin_win), zero)

    else:
        # not implemented yet
        result_real = op.Identity(x)
        result_imag = op.Identity(x)
        
    # final step, needs to move to first axis into the last position.
    result = op.Concat(result_real, result_imag, axis=0)
    if dim == one:
        final = op.Identity(result)
    else:
        if dim == two:
            final = op.Transpose(result, perm=[1, 0])
        else:
            if dim == three:
                final = op.Transpose(result, perm=[1, 2, 0])
            else:
                # It does not work for more than 4 dimensions.
                # The runtime fails here in that case due to an inconcistency
                # between result dimension and permutation size.
                final = op.Transpose(result, perm=[1, 2, 0])

    return final


if __name__ == "__main__":
    import numpy as np
    x = np.arange(5).astype(np.float32).reshape((1, -1))
    le = np.array([6], dtype=np.int64)
    print(dft(x, le))
