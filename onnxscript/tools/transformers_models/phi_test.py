# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=not-callable

import copy
import sys
import unittest

import numpy as np
import onnxruntime
import torch

import onnxscript.optimizer
import onnxscript.rewriter
import onnxscript.tools.training_helper
import onnxscript.tools.transformers_models
import onnxscript.tools.transformers_models.phi
from onnxscript._internal.version_utils import (
    has_transformers,
    ignore_warnings,
    torch_older_than,
)


class TestExportPhi(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.6"), reason="fails to export")
    @ignore_warnings(UserWarning)
    def test_phi_export_cpu(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.6"), reason="fails to export")
    @ignore_warnings(UserWarning)
    def test_phi_export_cpu_export_api(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(
            model, *input_tensors, export_api=True
        )
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @ignore_warnings(UserWarning)
    def test_phi_export_cuda(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors_cpu = input_tensors_many[0]
        model = model.to("cuda")
        input_tensors = [i.to("cuda") for i in input_tensors_cpu]
        expected = model(*input_tensors)
        proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.detach().cpu().numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @ignore_warnings(UserWarning)
    def test_phi_dort_static(self):
        model, input_tensors_many, _ = onnxscript.tools.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors_many[0]

        local_aot_ort = onnxscript.tools.training_helper.make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        expected_results, expected_gradients = onnxscript.tools.training_helper.train_loop(
            model, *input_tensors
        )
        results, gradients, onnx_models = onnxscript.tools.training_helper.train_loop(
            compiled_model,
            *input_tensors,
            dump_onnx_models=True,
            dump_prefix="_dump_dort_phi",
            dump_clean_first=True,
        )
        torch.testing.assert_allclose(expected_results[0], results[0], atol=1e-5, rtol=1e-5)

        # Checking there is only two generated graphs otherwise, it means there are graph breaks.
        self.assertEqual(len(onnx_models), 2)
        torch.testing.assert_allclose(
            expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
