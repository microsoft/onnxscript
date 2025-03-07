# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=not-callable

import sys
import unittest

import numpy as np
import onnxruntime
import torch

import onnxscript.optimizer
import onnxscript.rewriter
import onnxscript.tools.training_helper
import onnxscript.tools.transformers_models
import onnxscript.tools.transformers_models.mistral
from onnxscript._internal.version_utils import (
    has_transformers,
    ignore_warnings,
    torch_older_than,
    transformers_older_than,
)


class TestExportMistral(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @unittest.skipIf(not has_transformers(), reason="transformers is missing")
    @unittest.skipIf(torch_older_than("2.4"), reason="fails to export")
    @unittest.skipIf(
        transformers_older_than("4.42"), reason="cannot mutate tensors with frozen storage"
    )
    @ignore_warnings(UserWarning)
    def test_mistral_export_cpu(self):
        model, input_tensors_many, _ = (
            onnxscript.tools.transformers_models.mistral.get_mistral_model()
        )
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        try:
            proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        except torch._export.verifier.SpecViolationError as e:  # pylint: disable=protected-access
            # see https://github.com/pytorch/pytorch/issues/128394
            if "Node.meta _enter_autocast is missing val field." in str(e):
                raise unittest.SkipTest(str(e))
            raise
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
    @unittest.skipIf(torch_older_than("2.5"), reason="fails to export")
    @unittest.skipIf(
        transformers_older_than("4.42"), reason="cannot mutate tensors with frozen storage"
    )
    @ignore_warnings(UserWarning)
    def test_mistral_export_cpu_export_api(self):
        model, input_tensors_many, _ = (
            onnxscript.tools.transformers_models.mistral.get_mistral_model()
        )
        input_tensors = input_tensors_many[0]
        expected = model(*input_tensors)
        try:
            proto = onnxscript.tools.transformers_models.export_to_onnx(
                model, *input_tensors, export_api=True
            )
        except torch._export.verifier.SpecViolationError as e:  # pylint: disable=protected-access
            # see https://github.com/pytorch/pytorch/issues/128394
            if "Node.meta _enter_autocast is missing val field." in str(e):
                raise unittest.SkipTest(str(e))
            raise
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
        model, input_tensors_many, _ = (
            onnxscript.tools.transformers_models.mistral.get_mistral_model()
        )
        input_tensors_cpu = input_tensors_many[0]
        model = model.to("cuda")
        input_tensors = [i.to("cuda") for i in input_tensors_cpu]
        expected = model(*input_tensors)
        try:
            proto = onnxscript.tools.transformers_models.export_to_onnx(model, *input_tensors)
        except torch._export.verifier.SpecViolationError as e:  # pylint: disable=protected-access
            # see https://github.com/pytorch/pytorch/issues/128394
            if "Node.meta _enter_autocast is missing val field." in str(e):
                raise unittest.SkipTest(str(e))
            raise
        names = [i.name for i in proto.graph.input]
        np_input_tensors = [x.detach().cpu().numpy() for x in input_tensors]
        feeds = dict(zip(names, np_input_tensors))
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        results = sess.run(None, feeds)
        np.testing.assert_allclose(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
