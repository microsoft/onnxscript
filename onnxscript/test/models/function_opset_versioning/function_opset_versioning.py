# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16
from onnxscript.test.checker import isomorphic

import unittest

# opset_import:
# model opset import lists all opset that are used in the model
# function opset import lists all opset that are used by a function 
# one functions are nested, opset import is a superset of opset imports
# of nested function and opset import of itself. 
# The question is: 
# - how to specify version of a function: a function of different versions are defined in deparate files so there is no name clashing.
# - Model meta data are validated so that there is no version conflict (same domain but different version is not allowed)
# - how opset import meta data is used? How does runtime knows it can run the model given a model's opset import?    
# (1) how do we use Model meta data to capture this. how do we design libProto  
# (2) libProto: contains a list of functionProtos, all with the same domain. Because functionProto does not
# hold its opset version, libProto shall keep a version number - this means alll functionProtos shared the same
# version as well.
# (3) A question is do we want to libProto to contain more than just a list of functions that share the same domain and version?
# (4) I assume that a user will use a helper function to, given a python script file, "compile" it into a libProto.
# That means we shall force users not put multiple versions of a function into a file - which is a likely good pratice -
# otherwise there will be a function name duplication conflict.     
# (5) the most difficult thing is that a function is defined with a set opset import, it may conflict with
# opset import of the model that uses the function. this certainly is to happen if the function are not updated with
# the opset version for the model (at the model creation time by a converter, for example).
# To solve this, the version convert can be used when loading a libProto. 
# 

import onnx

class TestOnnxFunctionVersioning(unittest.TestCase):

    def test_model_with_function_versioning(self):
        from onnxscript.test.models.function_opset_versioning.Selu1 import Selu
        from onnxscript.test.models.function_opset_versioning.Elu1 import Elu
        selu_function_proto = Selu.function_ir.to_function_proto("")
        elu_function_proto = Elu.function_ir.to_function_proto("")
        selu_node = onnx.helper.make_node('Selu', ['i0'], ['o0'], domain='this')
        elu_node = onnx.helper.make_node('Elu', ['i1'], ['o1'], domain='this')
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=2, shape=[5])

        i0_value_info = onnx.helper.make_value_info(name='i0', type_proto=tensor_type_proto)
        i1_value_info = onnx.helper.make_value_info(name='i1', type_proto=tensor_type_proto)
        o0_value_info = onnx.helper.make_value_info(name='o0', type_proto=tensor_type_proto)
        o1_value_info = onnx.helper.make_value_info(name='o1', type_proto=tensor_type_proto)

        graph = onnx.helper.make_graph(
            [selu_node, elu_node],
            "a_graph",
            [i0_value_info, i1_value_info],
            [o0_value_info, o1_value_info])

        model = onnx.helper.make_model(
            graph,
            functions=[selu_function_proto, elu_function_proto],
            producer_name='test_versioning',
            opset_imports=[onnx.helper.make_opsetid("", 15), onnx.helper.make_opsetid("this", 1)])
        onnx.checker.check_model(model)

        lib_proto = onnx.helper.make_lib_proto(
            [selu_function_proto, elu_function_proto],
            producer_name='p2o')

        file_path = "C:/temp/selu_and_elu.libproto"
        onnx.save_lib_proto(lib_proto, file_path)

        lib_proto_loaded = onnx.load_lib_proto(file_path)

        model_from_lib_proto = onnx.helper.make_model(
            graph,
            functions=lib_proto_loaded.functions,
            producer_name='test_versioning',
            opset_imports=[onnx.helper.make_opsetid("", 15), onnx.helper.make_opsetid("this", 1)])
        onnx.checker.check_model(model_from_lib_proto)
        
        self.assertTrue(isomorphic(lib_proto_loaded.functions[0], selu_function_proto))
        self.assertTrue(isomorphic(lib_proto_loaded.functions[1], elu_function_proto))


if __name__ == '__main__':
    unittest.main()

import math
@script(Opset('this', 1))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    half = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=1/math.sqrt(2))
    one = op15.Constant(value_float=1.0)
    P1 = op15.MatMul(A, W)
    X = op15.Add(P1, Bias)
    T1 = op15.erf(X, b)
    T2 = op15.Add(one, T1)
    T3 = op15.Mul(half, T2)
    Y = op15.Mul(X, T3)
    return Y

@script(Opset('this', 1))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    half = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=1.0/math.sqrt(2.0))
    one = op15.Constant(value_float=1.0)

    X = op15.Gemm(A, W, Bias)

    T1 = op15.erf(X, b)
    T2 = op15.Add(one, T1)
    T3 = op15.Mul(half, T2)
    Y = op15.Mul(X, T3)
    return Y

@script(Opset('this', 2))
def gemmgelu(
        A: FLOAT[None],
        W: FLOAT[None],
        Bias: FLOAT[None]
) -> FLOAT[None]:
    a = op15.Constant(value_float=0.5)
    b = op15.Constant(value_float=0.797885)
    c = op15.Constant(value_float=0.035677)
    one = op15.Constant(value_float=1.0)

    X = op15.Gemm(A, W, Bias)

    T1 = op15.Mul(X, X)
    T2 = op15.Mul(c, T1)
    T3 = op15.Add(b, T2)
    T4 = op15.Mul(X, T3)
    T5 = op15.Tanh(T4)
    T6 = op15.Add(one, T5)
    T7 = op15.Mul(X, T6)
    Y = op15.Mul(a, T7)
    return Y
