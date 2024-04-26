"""Integration tests using Dynamo-ORT (DORT)."""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import unittest
import warnings
from typing import Callable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch._decomp
import torch._dynamo.backends.common
import torch.fx
import torch.onnx._internal.exporter
import transformers
import transformers.models.llama.modeling_llama
from torch.onnx._internal.diagnostics import infra as diagnostics_infra
from torch.onnx._internal.fx import (
    diagnostics,
    fx_onnx_interpreter,
    onnxfunction_dispatcher,
    passes,
)


@contextlib.contextmanager
def dump_onnx(prefix: str, *, folder: str, clean: bool = False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if clean:
        for f in os.listdir(folder):
            ff = os.path.join(folder, f)
            if os.path.isfile(ff):
                os.remove(ff)
    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    os.environ["ONNXRT_DUMP_PATH"] = os.path.join(folder, f"{prefix}_")

    try:
        yield
    finally:
        os.environ["ONNXRT_DUMP_PATH"] = value or ""


def make_aot_ort(dynamic: bool = False):
    ort_session_options = ort.SessionOptions()
    export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic)
    options = torch.onnx._OrtBackendOptions(
        export_options=export_options,
        ort_session_options=ort_session_options,
    )
    ort_backend = torch.onnx._OrtBackend(options=options)
    return ort_backend, ort_backend


def ids_tensor(shape, vocab_size):
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def _dynamo_export(
    graph_module,
    args,
    *,
    target_opset,
):
    context = diagnostics.DiagnosticContext(
        "_dynamo_export",
        torch.__version__,
        diagnostics_infra.DiagnosticOptions(),
    )
    onnx_registry = torch.onnx._internal.exporter.OnnxRegistry()

    function_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
        onnx_registry, context
    )

    graph_module = passes.MovePlaceholderToFront(context, graph_module).run()

    # Create the object to iterate through the nodes in graph one-by-one
    # and calls the corresponding ONNX exporter for each node.
    fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=context)
    # Cast FX variables if they will result schema-mismatch when searching
    # for ONNX operator. E.g., add(double_tensor, int_tensor) is fine in PyTorch,
    # but ONNX expects add(double_tensor, double_tensor).
    graph_module = passes.InsertTypePromotion(context, graph_module).run()

    # Start the per-node exporting process. It's conceptually a for loop
    # scanning through the nodes in the graph.
    exported = fx_interpreter.run(
        fx_graph_module=graph_module,
        onnxfunction_dispatcher=function_dispatcher,
        op_level_debug=False,
    )
    # Convert the exported result to ONNX ModelProto.
    onnx_model = exported.to_model_proto(opset_version=target_opset)

    # Modify ONNX model using pre-registered graph transforms.
    # They are in-place modifications for avoiding unnecessary
    # copy of ONNX initializers.
    return onnx_model


def get_decomposition_table():
    new_table = {}
    for k, v in torch._decomp.decomposition_table.items():
        if k.name() in {"aten::sigmoid_backward"}:
            new_table[k] = v
    return new_table


def common_llama_mixed_precision_small(folder_suffix: str, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    local_aot_ort, _ = make_aot_ort(dynamic=False)

    config = transformers.LlamaConfig(**kwargs)
    config._attn_implementation = "eager"

    model = transformers.models.llama.modeling_llama.LlamaModel(config).to("cuda")

    batch, seq, vocab_size = 2, 1024, 1024
    input_ids = ids_tensor([batch, seq], vocab_size).to("cuda")
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32)).to("cuda")

    model(input_ids, input_mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
            with dump_onnx(
                "dort-llama-ort",
                folder=f"dump_eager_llama_mixed_{folder_suffix}",
                clean=True,
            ):
                output = optimized_mod(input_ids, input_mask)
                output[0].sum().backward()

    names = [
        name
        for name in os.listdir(f"dump_eager_llama_mixed_{folder_suffix}")
        if name.endswith(".onnx")
    ]
    for name in names:
        model_proto = onnx.load(os.path.join(f"dump_eager_llama_mixed_{folder_suffix}", name))
        output_names = [o.name for o in model_proto.graph.output]

        for output_name in output_names:
            # This test fails if _unsafe_index_put is not supported, in that case,
            # DORT detects that a graph break is needed and let torch execute this instruction.
            # This is not desired.
            assert not output_name.startswith(
                "new_zeros"
            ), f"One output of the output is likely to be null, see {output_names}"


class TestCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    def test_llama(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        local_aot_ort, _ = make_aot_ort(dynamic=False)

        config = transformers.LlamaConfig(
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )
        config._attn_implementation = "eager"

        model = transformers.models.llama.modeling_llama.LlamaModel(config)

        batch, seq, vocab_size = 2, 1024, 1024
        input_ids = ids_tensor([batch, seq], vocab_size)

        model(input_ids)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
            with dump_onnx("dort-llama-ort", folder="dump_eager_llama", clean=True):
                output = optimized_mod(input_ids)  # , input_mask)
                output[0].sum().backward()

        names = [_ for _ in os.listdir("dump_eager_llama") if _.endswith(".onnx")]
        for name in names:
            onx = onnx.load(os.path.join("dump_eager_llama", name))
            output_names = [o.name for o in onx.graph.output]

            for o in output_names:
                # This test fails if _unsafe_index_put is not supported, in that case,
                # DORT detects that a graph break is needed and let torch execute this instruction.
                # This is not desired.
                assert not o.startswith(
                    "new_zeros"
                ), f"One output of the output is likely to be null, see {output_names}"

    @unittest.skipIf(not torch.cuda.is_available(), reason="not available on cpu")
    def test_llama_mixed_precision_small(self):
        common_llama_mixed_precision_small(
            "small",
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )

    @unittest.skipIf(not torch.cuda.is_available(), reason="not available on cpu")
    def test_llama_mixed_precision_large(self):
        # This test seems to produce a different model even though
        # the only difference is the model size. It might go through
        # a different code path in transformers.
        common_llama_mixed_precision_small(
            "large",
            hidden_size=4096,
            num_hidden_layers=1,
            vocab_size=32000,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
        )

    def test_mlp_dort(self):
        local_aot_ort, _ = make_aot_ort(dynamic=False)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        x = torch.randn(3, 10, dtype=torch.float32)

        model = MLP()
        expected = model(copy.deepcopy(x))
        expected.sum().backward()
        params = list(model.parameters())
        grad_expected = params[0].grad

        optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
        actual = optimized_mod(x)
        torch.testing.assert_close(expected, actual)

        actual.sum().backward()
        params2 = list(optimized_mod.parameters())
        grad_actual = params2[0].grad
        torch.testing.assert_close(grad_expected, grad_actual)

    def test_mlp_dort_custom_backend(self):
        def custom_backend(
            graph_module: torch.fx.GraphModule,
            args: Sequence[torch.Tensor],
            target_opset: int | None = None,
            use_cuda: bool = False,
        ) -> Callable:
            exported = _dynamo_export(graph_module, args, target_opset=target_opset)

            if use_cuda:
                providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
            else:
                providers = ("CPUExecutionProvider",)
            inference_session = ort.InferenceSession(
                exported.SerializeToString(), providers=providers
            )
            input_names = [i.name for i in inference_session.get_inputs()]

            def run(*args, **kwargs):
                assert len(kwargs) == 0, f"Not implemented with kwargs={kwargs}"
                # All inputs are moved to cpu with numpy. Not efficient but it simplifies
                # the unit tests.
                feeds = dict(zip(input_names, [x.detach().cpu().numpy() for x in args]))
                results = inference_session.run(None, feeds)
                return tuple(map(torch.Tensor, results))

            return run

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        x = torch.randn(3, 10, dtype=torch.float32)

        model = MLP()
        expected = model(copy.deepcopy(x))
        expected.sum().backward()
        params = list(model.parameters())
        grad_expected = params[0].grad

        aot_compiler = torch._dynamo.backends.common.aot_autograd(
            fw_compiler=lambda *args, **kwargs: custom_backend(
                *args, target_opset=18, **kwargs
            ),
            decompositions=get_decomposition_table(),
        )
        optimized_mod = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=False
        )

        actual = optimized_mod(x)
        torch.testing.assert_close(expected, actual)

        actual.sum().backward()
        params2 = list(optimized_mod.parameters())
        grad_actual = params2[0].grad
        torch.testing.assert_close(grad_expected, grad_actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
