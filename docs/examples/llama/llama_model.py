"""
Code modified from different sources:

* https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py
* https://github.com/pytorch/pytorch/pull/117009
* https://github.com/sdpython/experimental-experiment/blob/main/experimental_experiment/torch_helper/llama_helper.py
"""

import collections
import random
import time
from typing import Sequence, Tuple

import onnx
import onnx.inliner
import onnxruntime
import torch
import torch.export
import onnxscript
from onnxrewriter import optimizer
from onnxrewriter.rewriter import onnxruntime as ort_rewriter

from onnxscript.function_libs.torch_lib import _flags


def get_llama_decoder(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=1024,
    num_attention_heads=2,
    _attn_implementation="eager",
):
    """
    Returns the decoder part.
    See :func:`experimental_experiment.torch_helper.llama_helper.get_llama_model`.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    class LlamaDecoderWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.decoder = LlamaDecoderLayer(config, layer_idx=0)

        def forward(self, hidden_states, attention_mask, position_ids):
            (decoder_output,) = self.decoder(hidden_states, attention_mask, position_ids)
            return decoder_output

    def generate_example_inputs(batch: int, seq: int, hidden_size: int):
        # shape: batch x seq x hidden_size
        hidden_state = torch.randn(batch, seq, hidden_size)
        attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
        position_ids = torch.arange(0, seq, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).view(-1, seq)
        return hidden_state, attention_mask, position_ids

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, hidden_size))

    return LlamaDecoderWrapper(config), example_args_collection


def get_llama_attention(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=1024,
    num_attention_heads=2,
    _attn_implementation="eager",
):
    """
    Returns the attention part.
    See :func:`experimental_experiment.torch_helper.llama_helper.get_llama_model`.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    class LlamaAttentionWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attention = LlamaAttention(config, layer_idx=0)

        def forward(self, hidden_states, attention_mask, position_ids):
            attn_output, _, _ = self.attention(hidden_states, attention_mask, position_ids)
            return attn_output

    def generate_example_inputs(batch: int, seq: int, hidden_size: int):
        hidden_state = torch.randn(batch, seq, hidden_size)
        attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
        position_ids = torch.arange(0, seq, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).view(-1, seq)

        return hidden_state, attention_mask, position_ids

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, hidden_size))

    return LlamaAttentionWrapper(config), example_args_collection


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size

    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_llama_model(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=1024,
    num_attention_heads=2,
    _attn_implementation="eager",  # needed value to remove graph breaks
):
    """
    Returns a model.
    See `LlamaConfig
    <https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/llama2#transformers.LlamaConfig>`_.
    The parameters are chosen for a unit test configuration.
    For benchmark, a bigger one should be used.
    Commented out, the default value from :epkg:`transformers`.

    ::

        kwargs = dict(
            input_dims=[(2, 1024)] * 2,
            num_hidden_layers=1,  # 32
            hidden_size=512,  # 4096
            vocab_size=4000,  # 32000
            intermediate_size=2000,  # 11008
            max_position_embeddings=2048,
            num_attention_heads=8,  # 32
        )
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    class LlamaModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = LlamaModel(config)

        def forward(self, input_ids, attention_mask):
            model_output = self.model(input_ids, attention_mask=attention_mask)
            # Output 2, 3 are None
            return model_output[0], model_output[1]

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        input_ids = ids_tensor([batch, seq], vocab_size)
        input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
        assert input_mask.dtype == torch.float32
        return input_ids, input_mask

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    return LlamaModelWrapper(config), example_args_collection


def display_model_stats(model: onnx.ModelProto):
    """Displays the number of nodes, number of inputs, number of outputs and number of initializers."""
    print(f"number of nodes: {len(model.graph.node)}")
    unique_nodes = collections.Counter([node.op_type for node in model.graph.node])
    for k, v in unique_nodes.items():
        print(f"  {k}: {v}")
    print(f"number of inputs: {len(model.graph.input)}")
    print(f"number of outputs: {len(model.graph.output)}")
    print(f"number of initializers: {len(model.graph.initializer)}")
    print(f"number of functions: {len(model.functions)}")


def time_ort_model(model, name: str, example_args, run_count=100):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=options)
    input_data = [x.numpy() for x in example_args]
    start = time.time()
    ort_input = dict(zip((input.name for input in session.get_inputs()), input_data))
    for _ in range(run_count):
        session.run(None, ort_input)
    end = time.time()
    print(f"ORT Time {name}:", (end - start) / run_count, "average over", run_count, "runs")


def export():
    model, example_args_collection = get_llama_model(num_hidden_layers=10)

    exported = torch.export.export(model, example_args_collection[0])
    print("===exported fx graph===")
    print(exported)
    print("FX Node count:", len(exported.graph.nodes))
    exported_onnx = torch.onnx.dynamo_export(model, *example_args_collection[0]).model_proto
    print("===exported_onnx===")
    display_model_stats(exported_onnx)
    inlined_exported_onnx = onnx.inliner.inline_local_functions(exported_onnx)
    print("===inlined_exported_onnx===")
    display_model_stats(inlined_exported_onnx)

    _flags.EXPERIMENTAL_PREFER_TRACING = True

    exported_eager_onnx = torch.onnx.dynamo_export(
        model, *example_args_collection[0]
    ).model_proto
    print("===exported_eager_onnx===")
    display_model_stats(exported_eager_onnx)
    inlined_eager_exported_onnx = onnx.inliner.inline_local_functions(exported_eager_onnx)
    print("===inlined_eager_exported_onnx===")
    display_model_stats(inlined_eager_exported_onnx)
    # print(onnx.printer.to_text(inlined_eager_exported_onnx))
    onnx.save(inlined_eager_exported_onnx, "inlined_eager_exported_onnx.onnx")

    rewritten_model = optimizer.optimize(exported_onnx, num_iterations=2)
    # rewritten_model = ort_rewriter.rewrite(rewritten_model)
    rewritten_model = onnx.inliner.inline_local_functions(rewritten_model)
    onnx.save(rewritten_model, "rewritten_model.onnx")
    print("===rewritten_model===")
    display_model_stats(rewritten_model)

    rewritten_inlined_eager_exported_onnx = optimizer.optimize(
        exported_onnx, num_iterations=2, onnx_shape_inference=False
    )
    rewritten_inlined_eager_exported_onnx = onnx.inliner.inline_local_functions(
        rewritten_inlined_eager_exported_onnx
    )
    print("===rewritten_inlined_eager_exported_onnx===")
    display_model_stats(rewritten_inlined_eager_exported_onnx)
    rewritten_inlined_eager_exported_onnx2 = ort_rewriter.rewrite(
        rewritten_inlined_eager_exported_onnx
    )
    rewritten_inlined_eager_exported_onnx2 = onnx.inliner.inline_local_functions(
        rewritten_inlined_eager_exported_onnx2
    )
    print("===rewritten_inlined_eager_exported_onnx2===")
    display_model_stats(rewritten_inlined_eager_exported_onnx2)
    # print(onnx.printer.to_text(rewritten_inlined_eager_exported_onnx2))
    rewritten_inlined_eager_exported_onnx2 = onnx.shape_inference.infer_shapes(
        rewritten_inlined_eager_exported_onnx2, data_prop=True
    )
    onnx.save(
        rewritten_inlined_eager_exported_onnx2, "rewritten_inlined_eager_exported_onnx2.onnx"
    )

    torch.onnx.export(
        model, example_args_collection[0], "torchscript_model.onnx", opset_version=17
    )
    torchscript_model = onnx.load("torchscript_model.onnx")
    print("===torchscript_model===")
    display_model_stats(torchscript_model)

    # Time the model
    time_ort_model(inlined_exported_onnx, "inlined_exported_onnx", example_args_collection[0])
    time_ort_model(rewritten_model, "rewritten_model", example_args_collection[0])
    time_ort_model(
        inlined_eager_exported_onnx, "inlined_eager_exported_onnx", example_args_collection[0]
    )
    time_ort_model(
        rewritten_inlined_eager_exported_onnx2,
        "rewritten_inlined_eager_exported_onnx2",
        example_args_collection[0],
    )
    time_ort_model(torchscript_model, "torchscript_model", example_args_collection[0])

    # Time the model
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(*example_args_collection[0])
    end = time.time()
    print("Eager Time:", (end - start) / 100)


if __name__ == "__main__":
    export()
