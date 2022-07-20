# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import unittest
import os
import tempfile

import onnx
import onnx.parser
import onnx.version_converter
import onnxruntime as onnxrt

from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16
from onnxscript.test.checker import isomorphic

from onnxscript.test.models.function_opset_versioning.MyMulAdd1 import MyMul as MyMul15, MyAdd as MyAdd15, MyWhere as MyWhere15
from onnxscript.test.models.function_opset_versioning.MyMulAdd2 import MyMul as MyMul16, MyAdd as MyAdd16, MyWhere as MyWhere16

@script()
def MyAddTwiceModel1(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op15.Add(X, A)
    return MyAdd15(tmp, A)

@script()
def MyAddTwiceModel2(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op16.Add(X, A)
    return MyAdd16(tmp, A)

@script()
def MyAddTwiceModelMix(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op16.Add(X, A)
    return MyAdd15(tmp, A)

@script()
def MyAddWhereModelMix(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op16.Add(X, A)
    return MyWhere15(tmp)

@script()
def MyAddWhereModelMix2(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op15.Add(X, A)
    return MyWhere16(tmp)

@script()
def MyAddTwiceModel1(X: FLOAT['N'], A: FLOAT['N']) -> FLOAT['N']:
    tmp = op15.Add(X, A)
    return MyAdd15(tmp, A)

class TestOnnxFunctionVersioning2(unittest.TestCase):
    def test_simple_versioning(self, model_script):
        model = model_script.to_model_proto()
        onnx.checker.check_model(model)
        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())
        result = sess.run(
            None,
            {
                "X": np.random.rand(2).astype(np.float32),
                "A": np.random.rand(2).astype(np.float32),
            })
        print(result)

    def test_lib_proto_save_load(self):
        lib_proto = onnx.helper.make_lib_proto(
            [MyAdd15.to_function_proto(), MyWhere15.to_function_proto()],
            producer_name='my')
        onnx.save_lib_proto(lib_proto, "MyScriptLibVer1.libproto")

        lib_proto_loaded = onnx.load_lib_proto("MyScriptLibVer1.libproto")
        MyAdd15_proto, _ = lib_proto_loaded.functions

        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=1, shape=[2])
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node('Add', ['X', 'A'], ['tmp']),
                onnx.helper.make_node('MyAdd15', ['tmp', 'A'], ['Y'], domain='this')
            ],
            "a_graph",
            [
                onnx.helper.make_value_info(name='X', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='A', type_proto=tensor_type_proto),
            ],
            [onnx.helper.make_value_info(name='Y', type_proto=tensor_type_proto)])

        model = onnx.helper.make_model(
            graph,
            functions=[MyAdd15_proto],
            producer_name='my',
            opset_imports=[onnx.helper.make_opsetid("", 15), onnx.helper.make_opsetid("this", 1)])

        onnx.checker.check_model(model)

    def test_version_converter(self):
        input = '''
            <
            ir_version: 9,
            opset_import: [ "" : 15, "custom_domain" : 1],
            producer_name: "FunctionProtoTest",
            producer_version: "1.0",
            model_version: 1,
            doc_string: "A test model for model local functions."
          >
         agraph (float[N] x) => (float[N] out)
         {
            out = custom_domain.Square(x)
         }

         <
         domain: "custom_domain",
         opset_import: [ "" : 15],
         doc_string: "Test function proto"
         >
           Square
           (X) => (C)
           {
               C = Mul(X, X)
           }
         '''

        model = onnx.parser.parse_model(input)
        onnx.checker.check_model(model)


        model = MyAddWhereModelMix2.to_model_proto()
        print(model)
        model_converted = onnx.version_converter.convert_version(model=model, target_version=16)
        print(model_converted)

    def test_version(self):

        self.test_simple_versioning(MyAddTwiceModel1)
        self.test_simple_versioning(MyAddTwiceModel2)
        self.test_simple_versioning(MyAddTwiceModelMix)
        self.test_simple_versioning(MyAddWhereModelMix)
        self.test_version_converter()

if __name__ == '__main__':
    unittest.main()
