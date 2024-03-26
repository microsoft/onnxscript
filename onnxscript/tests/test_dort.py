import contextlib
import copy
import os
import unittest
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import onnx

VERBOSE: int = 0


def has_cuda():
    import torch

    return torch.cuda.is_available()


@contextlib.contextmanager
def dump_onnx(prefix: str, folder: Optional[str] = None, clean: bool = False):
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
        if clean:
            for f in os.listdir(folder):
                ff = os.path.join(folder, f)
                if os.path.isfile(ff):
                    os.remove(ff)
    else:
        assert not clean, "cleaning can only happen if folder is specified"

    value = os.environ.get("ONNXRT_DUMP_PATH", None)
    os.environ["ONNXRT_DUMP_PATH"] = os.path.join(folder, f"{prefix}_")

    try:
        yield
    finally:
        os.environ["ONNXRT_DUMP_PATH"] = value or ""


def make_aot_ort(
    dynamic: bool = False,
    rewrite: bool = False,
    verbose: int = 0,
):
    import onnxruntime  # noqa: I001
    from torch.onnx import (
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    if rewrite:
        try:
            import onnxrewriter  # noqa: F401
        except ImportError:
            rewrite = False

    ort_session_options = onnxruntime.SessionOptions()
    export_options = ExportOptions(dynamic_shapes=dynamic)

    if rewrite:
        from onnxrewriter.optimizer import optimize
        from onnxrewriter.rewriter import rewrite

        def optimize_model_proto(model_proto):
            first_model_proto = model_proto
            model_proto = optimize(
                model_proto,
                num_iterations=2,
                onnx_shape_inference=False,
            )
            model_proto = rewrite(model_proto)
            del first_model_proto.graph.node[:]
            del first_model_proto.functions[:]
            first_model_proto.graph.node.extend(model_proto.graph.node)
            first_model_proto.functions.extend(model_proto.functions)
            return first_model_proto

        if verbose:
            print("[make_aot_ort] enable rewriting")

        options = OrtBackendOptions(
            export_options=export_options,
            ort_session_options=ort_session_options,
            pre_ort_model_transforms=[optimize_model_proto],
        )
    else:
        options = OrtBackendOptions(
            export_options=export_options,
            ort_session_options=ort_session_options,
        )

    ort_backend = OrtBackend(options=options)
    return ort_backend, ort_backend


def ids_tensor(shape, vocab_size):
    import torch

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def _extract_graph_module_outputs(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
) -> Any:
    """Collect "val" fields from outputs metadata in this torch.fx.GraphModule."""
    for node in graph_module.graph.nodes:
        if node.op == "output":
            # Output node is unique. Let's retrieve output values from
            # this node's input list. And then just return.
            return node.args[0]
    raise ValueError("No output node found in this torch.fx.GraphModule.")


def _maybe_map_to_meta_val(value):
    if hasattr(value, "meta") and "val" in value.meta:
        # Select outputs with "val" information. Without "val",
        # it's not possible access output_arg.meta["val"].device.
        return value.meta["val"]
    else:
        return value


def _dynamo_export(
    graph_module,
    args,
    verbose,
    target_opset,
    **kwargs,
):
    import torch
    from torch.onnx._internal.diagnostics import infra
    from torch.onnx._internal.exporter import OnnxRegistry
    from torch.onnx._internal.fx import (
        diagnostics,
        fx_onnx_interpreter,
        onnxfunction_dispatcher,
    )

    context = diagnostics.DiagnosticContext(
        "_dynamo_export",
        torch.__version__,
        infra.DiagnosticOptions(),
    )
    onnx_registry = OnnxRegistry()

    self_onnxfunction_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
        onnx_registry, context
    )

    graph_module = torch.onnx._internal.fx.passes.MovePlaceholderToFront(
        context, graph_module
    ).run()

    # Create the object to iterate through the nodes in graph one-by-one
    # and calls the corresponding ONNX exporter for each node.
    fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=context)
    # Cast FX variables if they will result schema-mismatch when searching
    # for ONNX operator. E.g., add(double_tensor, int_tensor) is fine in PyTorch,
    # but ONNX expects add(double_tensor, double_tensor).
    graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(
        context, graph_module
    ).run()

    # Start the per-node exporting process. It's conceptually a for loop
    # scanning through the nodes in the graph.
    exported = fx_interpreter.run(
        fx_graph_module=graph_module,
        onnxfunction_dispatcher=self_onnxfunction_dispatcher,
        op_level_debug=False,
    )
    # Convert the exported result to ONNX ModelProto.
    onnx_model = exported.to_model_proto(opset_version=target_opset)

    # Modify ONNX model using pre-registered graph transforms.
    # They are in-place modifications for avoiding unnecessary
    # copy of ONNX initializers.
    return onnx_model


def custom_backend(
    graph_module: "torch.fx.GraphModule",  # noqa: F821
    args: List["torch.Tensor"],  # noqa: F821
    target_opset: Optional[int] = None,
    verbose: Union[int, Tuple[int, int]] = 0,
    use_cuda: bool = False,
) -> Callable:
    import onnxruntime
    import torch

    onx = _dynamo_export(graph_module, args, verbose, target_opset)

    providers = ["CPUExecutionProvider"]
    if use_cuda:
        providers.insert(0, "CUDAExecutionProvider")
    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def run(*args, **kwargs):
        assert len(kwargs) == 0, f"Not implemented with kwargs={kwargs}"
        # All inputs are moved to cpu with numpy. Not efficient but it simplifies
        # the unit tests.
        feeds = dict(zip(input_names, map(lambda x: x.detach().cpu().numpy(), args)))
        results = sess.run(None, feeds)
        return tuple(map(torch.Tensor, results))

    return run


def get_decomposition_table():
    import torch

    new_table = {}
    for k, v in torch._decomp.decomposition_table.items():
        if k.name() in {"aten::sigmoid_backward"}:
            new_table[k] = v
    return new_table


class TestCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime  # noqa: F401

    def test_llama(self):
        import torch
        import torch.onnx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaModel

        local_aot_ort, _ = make_aot_ort(
            dynamic=False,
            rewrite=True,
            verbose=1,
        )

        config = LlamaConfig(
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )
        config._attn_implementation = "eager"

        model = LlamaModel(config)  # .to("cuda")

        batch, seq, vocab_size = 2, 1024, 1024
        input_ids = ids_tensor([batch, seq], vocab_size)  # .to("cuda")
        # input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

        model(input_ids)  # , input_mask)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
            with dump_onnx("dort-llama-ort", folder="dump_eager_llama", clean=True):
                output = optimized_mod(input_ids)  # , input_mask)
                output[0].sum().backward()

        names = [_ for _ in os.listdir("dump_eager_llama") if _.endswith(".onnx")]
        if VERBOSE:
            print("------------------------------------------")
            print(f"exported model: {names}")
        for name in names:
            if VERBOSE:
                print()
                print(f"NODES in {name!r}")
            onx = onnx.load(os.path.join("dump_eager_llama", name))
            if VERBOSE:
                for i, node in enumerate(onx.graph.node):
                    print(
                        f"{i+1}/{len(onx.graph.node)}: "
                        f"{node.op_type} {node.input} -> {node.output}"
                    )
            output_names = [o.name for o in onx.graph.output]

            for o in output_names:
                # This test fails if _unsafe_index_put is not supported, in that case,
                # DORT detects that a graph break is needed and let torch execute this instruction.
                # This is not desired.
                assert not o.startswith(
                    "new_zeros"
                ), f"One output of the output is likely to be null, see {output_names}"

    def common_llama_mixed_precision_small(self, folder_suffix, **kwargs):
        import torch
        import torch.onnx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaModel

        local_aot_ort, _ = make_aot_ort(dynamic=False, rewrite=True, verbose=1)

        config = LlamaConfig(**kwargs)
        config._attn_implementation = "eager"

        model = LlamaModel(config).to("cuda")

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
            _
            for _ in os.listdir(f"dump_eager_llama_mixed_{folder_suffix}")
            if _.endswith(".onnx")
        ]
        if VERBOSE:
            print("------------------------------------------")
            print(f"exported model: {names}")
        for name in names:
            if VERBOSE:
                print()
                print(f"NODES in {name!r}")
            onx = onnx.load(os.path.join(f"dump_eager_llama_mixed_{folder_suffix}", name))
            if VERBOSE:
                for i, node in enumerate(onx.graph.node):
                    print(
                        f"{i+1}/{len(onx.graph.node)}: "
                        f"{node.op_type} {node.input} -> {node.output}"
                    )
            output_names = [o.name for o in onx.graph.output]

            for o in output_names:
                # This test fails if _unsafe_index_put is not supported, in that case,
                # DORT detects that a graph break is needed and let torch execute this instruction.
                # This is not desired.
                assert not o.startswith(
                    "new_zeros"
                ), f"One output of the output is likely to be null, see {output_names}"

    @unittest.skipIf(not has_cuda(), reason="not available on cpu")
    def test_llama_mixed_precision_small(self):
        self.common_llama_mixed_precision_small(
            "small",
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )

    @unittest.skipIf(not has_cuda(), reason="not available on cpu")
    def test_llama_mixed_precision_large(self):
        self.common_llama_mixed_precision_small(
            "large",
            hidden_size=4096,
            num_hidden_layers=1,
            vocab_size=32000,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
        )

    def test_mlp_dort(self):
        import torch
        import torch.onnx

        local_aot_ort, _ = make_aot_ort(dynamic=False, rewrite=True, verbose=1)

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
        got = optimized_mod(x)
        assert torch.allclose(expected, got)

        got.sum().backward()
        params2 = list(optimized_mod.parameters())
        grad_got = params2[0].grad
        assert torch.allclose(grad_expected, grad_got)

    def test_mlp_dort_custom_backend(self):
        import torch
        import torch.onnx
        from torch._dynamo.backends.common import aot_autograd

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

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: custom_backend(
                *args, target_opset=18, **kwargs
            ),
            decompositions=get_decomposition_table(),
        )
        optimized_mod = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=False
        )

        got = optimized_mod(x)
        assert torch.allclose(expected, got)

        got.sum().backward()
        params2 = list(optimized_mod.parameters())
        grad_got = params2[0].grad
        assert torch.allclose(grad_expected, grad_got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
