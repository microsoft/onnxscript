import copy
import sys
import unittest

import numpy as np
import onnx.inliner
import onnxruntime
import torch

import onnxscript.optimizer
import onnxscript.rewriter
import onnxscript.testing.training_helper
import onnxscript.testing.transformers_models
import onnxscript.testing.transformers_models.phi

HAS_TRANSFORMERS = onnxscript.testing.transformers_models.has_transformers()


def export_to_onnx(model, *input_tensors, optimize=True):
    prog = torch.onnx.dynamo_export(model, *input_tensors)
    model_proto = prog.model_proto
    if optimize:
        model_proto = onnxscript.optimizer.optimize(
            model_proto,
            num_iterations=2,
            onnx_shape_inference=True,
        )
        model_proto = onnxscript.rewriter.rewrite(model_proto)
        model_proto = onnx.inliner.inline_local_functions(model_proto)
    return model_proto


class TestExportPhi(unittest.TestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not HAS_TRANSFORMERS, reason="transformers is missing")
    def test_phi_export_cpu(self):
        model, input_tensors = onnxscript.testing.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)
        proto = export_to_onnx(model, *input_tensors)
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
    @unittest.skipIf(not HAS_TRANSFORMERS, reason="transformers is missing")
    def test_phi_export_cuda(self):
        model, input_tensors = onnxscript.testing.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors[0]
        model = model.to("cuda")
        input_tensors = [i.to("cuda") for i in input_tensors]
        expected = model(*input_tensors)
        proto = export_to_onnx(model, *input_tensors)
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.detach().cpu().numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not HAS_TRANSFORMERS, reason="transformers is missing")
    def test_phi_dort_static(self):
        model, input_tensors = onnxscript.testing.transformers_models.phi.get_phi_model()
        input_tensors = input_tensors[0]
        expected = model(*input_tensors)

        local_aot_ort = onnxscript.testing.training_helper.make_aot_ort(dynamic=False)

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=local_aot_ort,
            dynamic=False,
            fullgraph=True,
        )

        results = compiled_model(*input_tensors)
        torch.testing.assert_allclose(expected[0], results[0], atol=1e-5, rtol=1e-5)

        expected_gradients = onnxscript.testing.training_helper.train_loop(
            model, *input_tensors
        )
        gradients = onnxscript.testing.training_helper.train_loop(
            compiled_model, *input_tensors
        )
        torch.testing.assert_allclose(
            expected_gradients[0], gradients[0], atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
