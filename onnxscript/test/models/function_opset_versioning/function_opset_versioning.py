# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
import unittest
import os
import tempfile
from onnxscript import script
from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.values import Opset
from onnxscript.onnx import opset15 as op15
from onnxscript.onnx import opset16 as op16
from onnxscript.test.checker import isomorphic

class TestOnnxFunctionVersioning(unittest.TestCase):
    def build_test_model_with_lib_proto(self, selu, elu):
        selu_function_proto = selu.function_ir.to_function_proto()
        elu_function_proto = elu.function_ir.to_function_proto()
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

    def test_model_with_function_versioning_ver1(self):
        from onnxscript.test.models.function_opset_versioning.Selu1 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu1 import Elu as elu
        self.build_test_model_with_lib_proto(selu, elu)

    def test_model_with_function_versioning_ver2(self):
        from onnxscript.test.models.function_opset_versioning.Selu2 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu2 import Elu as elu
        self.build_test_model_with_lib_proto(selu, elu)

    def test_model_with_function_versioning_mixed_vers(self):
        from onnxscript.test.models.function_opset_versioning.Selu1 import Selu as selu
        from onnxscript.test.models.function_opset_versioning.Elu2 import Elu as elu
        # TODO: checker shall fail if there is a conflict of opset_import version.
        # TODO: irbuilder shall used version converter to convert onnx ops if there is a conflict. 
        self.build_test_model_with_lib_proto(selu, elu)


if __name__ == '__main__':
    unittest.main()
