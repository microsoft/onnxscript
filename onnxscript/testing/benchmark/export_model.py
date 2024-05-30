# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import textwrap

import onnxscript.testing.benchmark

args = onnxscript.testing.benchmark.get_parsed_args(
    "export_model",
    description=textwrap.dedent(
        """Measures the inference time for a particular model.
        This script can be used to quickly evaluate the improvment made by a pattern optimization
        for a particular model.

        Example::

            python -m onnxscript.testing.benchmark.export_model --model phi --device cuda --config large --num_hidden_layers=10 --mixed=1 --dynamic=0 --exporter=dynamo
        """
    ),
    repeat=("10", "number of inferences to measure"),
    warmup=("5", "number of inferences to warm"),
    model=("phi", "model to measure, llama, mistral, phi, ..."),
    exporter=("dynamo", "script, dynamo"),
    device=("cpu", "'cpu' or 'cuda'"),
    target_opset=(18, "opset to convert into, use with backend=custom"),
    config=("default", "default, medium, or small to test"),
    verbose=(0, "verbosity"),
    dump_folder=("", "if not empty, dump the model in that folder"),
    dump_ort=(1, "produce the model optimized by onnxruntime"),
    optimize=(1, "optimize the model"),
    ort_optimize=(1, "enable or disable onnxruntime optimization"),
    mixed=(0, "mixed precision (based on autocast)"),
    dynamic=("0", "use dynamic shapes"),
    num_hidden_layers=(1, "number of hidden layers"),
    with_mask=(1, "with or without mask, dynamo may fail with a mask"),
)

print("-------------------")
print("Evaluate with model {args.model!r}")
print(args)
print("-------------------")
