from onnxscript.onnx_types import FLOAT, INT64


def dummy_tensor(N: INT64[1]) -> FLOAT["N", "N"]:

    shape = oxs.Shape(N)
    one = oxs.Reshape(oxs.Constant(value_floats=[1.]), shape)
    zeroi = oxs.Reshape(oxs.Constant(value_ints=[0]), shape)
    onei = oxs.Reshape(oxs.Constant(value_ints=[1]), shape)
    minusi = oxs.Neg(onei)  # oxs.Constant(value_int64=-1) fails
    shape1 = oxs.Concat(minusi, onei, axis=0)  # oxs.Constant(value_floats=[-1, 1])  fails

    nar = oxs.Range(zeroi, N, onei)  
    n0 = oxs.Cast(nar, to=1)
    n = oxs.Reshape(n0, shape1)
    nt = oxs.Transpose(n, perm=[1, 0])

    p = nt * n
    return p


if __name__ == '__main__':
    import numpy as np
    from numpy.testing import assert_almost_equal
    from onnxscript import eager_mode_evaluator as oxs
    n = np.array([3], dtype=np.int64)
    result = dummy_tensor(n)
    expected = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4]], dtype=np.float32)
    assert_almost_equal(expected, result)
    print("done")
