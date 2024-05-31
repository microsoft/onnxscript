# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Sequence

import onnx


def get_parsed_args(
    name: str,
    scenarios: dict[str, str] | None = None,
    description: str | None = None,
    epilog: str | None = None,
    new_args: list[str] | None = None,
    **kwargs: dict[str, tuple[int | str | float, str]],
) -> dict[str, Any]:
    """
    Returns parsed arguments for examples in this package.

    :param name: script name
    :param scenarios: list of available scenarios
    :param description: parser description
    :param epilog: text at the end of the parser
    :param number: default value for number parameter
    :param repeat: default value for repeat parameter
    :param warmup: default value for warmup parameter
    :param sleep: default value for sleep parameter
    :param expose: if empty, keeps all the parameters,
        if not None, only publish kwargs contains, otherwise the list
        of parameters to publish separated by a comma
    :param new_args: args to consider or None to take `sys.args`
    :param kwargs: additional parameters,
        example: `n_trees=(10, "number of trees to train")`
    :return: parser
    """
    parser = argparse.ArgumentParser(
        prog=name,
        description=description or f"Available options for {name}.py.",
        epilog=epilog or "",
    )
    for k, v in kwargs.items():
        parser.add_argument(
            f"--{k}",
            help=f"{v[1]}, default is {v[0]}",
            type=type(v[0]),
            default=v[0],
        )

    parsed = parser.parse_args(args=new_args)
    return {k: getattr(parsed, k) for k in kwargs}


def common_export(
    model: Any,
    inputs: Sequence[Any],
    exporter: str = "dynamo",
    target_opset: int = 18,
    folder: str = "",
    filename: str = "model.onnx",
    dynamic_shapes: Any | None = None,
    verbose: int = 0,
    optimization: str | None = None,
):
    """
    Exports a model into a folder.

    Args:
        model: model
        exporter: script, dynamo
        folder: folder to export into
        filename: onnx filename
        inputs: inputs
        dynamic_shapes: dynamic shapes
        target_opset: target opset
        optimization: optimization scenario
        verbose: verbosity

    Returns:
        onnx proto

    """
    import torch.onnx

    if folder:
        if not os.path.exists(folder):
            os.mkdir(folder)
        filename = os.path.join(folder, filename)

    if verbose:
        print(f"[common_export] start exporting with {exporter!r} in {filename!r}")
    begin = time.perf_counter()
    if exporter == "script":
        torch.onnx.export(
            model,
            inputs,
            filename,
            do_constant_folding=False,
            input_names=[f"input{i}" for i in range(len(inputs))],
            opset_version=target_opset,
        )
    elif exporter == "dynamo":
        with torch.no_grad():
            prog = torch.onnx.dynamo_export(model, *inputs)
        onx = prog.model_proto
        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
    else:
        raise AssertionError(f"Unknown exporter {exporter!r}")

    if verbose:
        print(f"[common_export] exporter done in {time.perf_counter() - begin}s")
        print(f"[common_export] size of the export: {os.stat(filename).st_size / 2**20} Mb")

    with open(filename, "rb") as f:
        onx = onnx.load(filename)

    if optimization:
        onx = optimize_model_proto(onx, optimization, verbose=verbose)
        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())

    return onx


def optimize_model_proto(
    model_proto: onnx.ModelProto, optimization: str | None = None, verbose: int = 0
):
    """
    Optimizes a model given some scenarios.

    Args:
        model_proto: ModelProto
        optimization: comma separated value
        verbose: verbosity

    Returns:
        optmized model
    """
    if not optimization:
        return model_proto

    for value in optimization.split(","):

        if verbose:
            print(f"[optimize_model_proto] start {value}")
            begin = time.perf_counter()

        if value == "optimize":

            import onnxscript.optimizer

            model_proto = onnxscript.optimizer.optimize(
                model_proto,
                num_iterations=2,
                onnx_shape_inference=False,
            )

        elif value == "rewrite":
            import onnxscript.rewriter

            model_proto = onnxscript.rewriter.rewrite(model_proto)

        elif value == "inline":
            import onnx.inliner

            model_proto = onnx.inliner.inline_local_functions(model_proto)

        else:
            raise AssertionError(
                f"Optimization step {value!r} is not implemented in {optimization!r}"
            )

        if verbose:
            print(f"[optimize_model_proto] {value} done in {time.perf_counter() - begin}")

    return model_proto


def run_inference(
    model: Any,
    example_inputs: Sequence[Any],
    warmup: int = 5,
    repeat: int = 5,
    verbose: int = 0,
) -> dict[str, Any]:
    """
    Runs multiple times the same inference.

    Args:
        model: torch model to run
        example_inputs: dummy inputs
        warmup: number of iterations to warmup
        repeat: number of iterations to repeat
        verbose: verbosity

    Returns:
        statistcs
    """
    if verbose:
        print(f"[run_inference] start {warmup} warmup iterations")

    stats = {}
    iterations = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        model(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["warmup"] = warmup
    stats["warmup_time"] = end
    stats["warmup_iter"] = iterations

    if verbose:
        print(f"[run_inference] warmup done in {time.perf_counter() - begin}")
        print(f"[run_inference] start {repeat} iterations")

    iterations = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        model(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["repeat"] = repeat
    stats["repeat_time"] = end
    stats["repeat_iter"] = iterations

    if verbose:
        print(f"[run_inference] measure done in {time.perf_counter() - begin}")

    return stats
