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
import onnxruntime as onnxrt

from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16
from onnxscript.test.checker import isomorphic

from onnxscript.test.models.function_opset_versioning.Selu2 import Selu as Selu16
from onnxscript.test.models.function_opset_versioning.Selu1 import Selu as Selu15
from onnxscript.test.models.function_opset_versioning.Elu1 import Elu as Elu15


@script()
def ScriptModelWithVersionConflict(i0: FLOAT['N'], i1: FLOAT['N']) -> FLOAT['N']:
    o0 = Selu16(i0)
    o1 = Elu15(i1)
    return o0 + o1

@script()
def ScriptModelWithoutVersionConflict(i0: FLOAT['N'], i1: FLOAT['N']) -> FLOAT['N']:
    o0 = Selu15(i0)
    o1 = Elu15(i1)
    return o0 + o1

class TestOnnxFunctionVersioning(unittest.TestCase):
    def test_script_model(self):
        model = ScriptModelWithoutVersionConflict.function_ir.to_model_proto()
        onnx.checker.check_model(model)
        print(model)

        i0 = np.random.rand(2).astype(np.float32)
        i1 = np.random.rand(2).astype(np.float32)

        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())

        result = sess.run([], {"i0": i0, "i1": i1})
        print(result)

    def build_test_model_with_lib_proto(self, selu, elu):
        '''
        build a model with Selu and Elu function op
        Elu in turn uses Selu
        The 2 function op may not share the same ONNXAI opset version.
        In such case, checker shall raise an error
        version converter shall be used so that there is consistent opsert version per domain 
        '''
        selu_function_proto = selu.function_ir.to_function_proto()
        elu_function_proto = elu.function_ir.to_function_proto()
        selu_node = onnx.helper.make_node('Selu', ['i0'], ['o0'], domain='this')
        elu_node = onnx.helper.make_node('Elu', ['i1'], ['o1'], domain='this')
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=1, shape=[5])

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

        lib_proto_path = os.path.join(tempfile.mkdtemp(), "selu_and_elu.libproto")
        onnx.save_lib_proto(lib_proto, lib_proto_path)

        lib_proto_loaded = onnx.load_lib_proto(lib_proto_path)

        model_from_lib_proto = onnx.helper.make_model(
            graph,
            functions=lib_proto_loaded.functions,
            producer_name='test_versioning',
            opset_imports=[onnx.helper.make_opsetid("", 15), onnx.helper.make_opsetid("this", 1)])
        onnx.checker.check_model(model_from_lib_proto)
        
        self.assertTrue(isomorphic(lib_proto_loaded.functions[0], selu_function_proto))
        self.assertTrue(isomorphic(lib_proto_loaded.functions[1], elu_function_proto))

        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())

    def test_model_with_function_versioning_ver1(self):
        from onnxscript.test.models.function_opset_versioning.Selu1 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu1 import Elu as elu
        self.build_test_model_with_lib_proto(selu, elu)

    def test_model_with_function_versioning_ver2(self):
        from onnxscript.test.models.function_opset_versioning.Selu2 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu2 import Elu as elu
        self.build_test_model_with_lib_proto(selu, elu)

    def test_model_with_function_versioning_mixed_vers(self):
        from onnxscript.test.models.function_opset_versioning.Selu2 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu1 import Elu as elu
        # TODO: checker shall fail if there is a conflict of opset_import version.
        # TODO: irbuilder shall used version converter to convert onnx ops if there is a conflict. 
        self.build_test_model_with_lib_proto(selu, elu)

    def test_simple_versioning(self):
        from onnxscript.test.models.function_opset_versioning.Elu1 import MyMul, MyAdd, MyMulAdd
        @script()
        def MyModel1(X: FLOAT['N'], A: FLOAT['N'], B: FLOAT['N']) -> FLOAT['N']:
            tmp = MyMul(X, A)
            return MyAdd(tmp, B)

        model = MyModel1.to_model_proto()
        onnx.checker.check_model(model)
        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())
        result = sess.run(
            None,
            {
                "X": np.random.rand(2).astype(np.float32),
                "A": np.random.rand(2).astype(np.float32),
                "B": np.random.rand(2).astype(np.float32)
            })
        print(result)


    def test_model_script_matmul_with_optional_inputs(self):
        model_script = '''
            <
            ir_version: 7,
            opset_import: [ "" : 13 ]
            >
            MatMul (float[2] X, float[2,3] W, bool b, float[3] Bias) => (float[3] Z)
            {
            Z = If (b) <
                then_branch = g1 () => (float[3] z_then) {
                    tmp = MatMul(X, W)
                    z_then = Add(tmp, Bias)
                    },
                else_branch = g2 () => (float[3] z_else) { z_else = MatMul(X, W) }
                >
            }
           '''
        model = onnx.parser.parse_model(model_script)
        print(model)

        x = np.random.rand(2).astype(np.float32)
        w = np.random.rand(2, 3).astype(np.float32)
        bias = np.random.rand(3).astype(np.float32)
        b = np.array([0]).astype(bool)

        sess = onnxrt.InferenceSession(model.SerializeToString(), providers=onnxrt.get_available_providers())

        result_no_bias = sess.run([], {"X": x, "W": w, "Bias": bias, "b": b})
        print(result_no_bias) 

        b = np.array([1]).astype(bool)
        result_with_bias = sess.run([], {"X": x, "W": w, "Bias": bias, "b": b})
        print(result_with_bias) 

        print("")


if __name__ == '__main__':
    unittest.main()
