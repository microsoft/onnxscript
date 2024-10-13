# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=import-outside-toplevel, no-else-raise, consider-using-with, consider-using-enumerate

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import onnx
import onnx.inliner

import onnxscript.optimizer
import onnxscript.rewriter
import onnxscript.rewriter.llama_rule_sets as rules
import onnxscript.rewriter.onnxruntime as ort_rules
import onnxscript.rewriter.pattern as orp
from onnxscript import ir
from onnxscript.optimizer._remove_unused import remove_unused_nodes


def get_parsed_args(
    name: str,
    description: str | None = None,
    epilog: str | None = None,
    new_args: list[str] | None = None,
    **kwargs: tuple[Any, str],
) -> dict[str, Any]:
    """
    Returns parsed arguments for examples in this package.

    Args:
        name: script name
        scenarios: list of available scenarios
        description: parser description
        epilog: text at the end of the parser
        number: default value for number parameter
        repeat: default value for repeat parameter
        warmup: default value for warmup parameter
        sleep: default value for sleep parameter
        expose: if empty, keeps all the parameters,
            if not None, only publish kwargs contains, otherwise the list
            of parameters to publish separated by a comma
        new_args: args to consider or None to take `sys.args`
        kwargs: additional parameters,
            example: `n_trees=(10, "number of trees to train")`

    Returns:
        interpreted parameters in a dictionary
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


class BenchmarkError(RuntimeError):
    pass


def get_machine() -> dict[str, Any]:
    """Returns the machine specification."""
    cpu: dict[str, Any] = dict(
        machine=str(platform.machine()),
        processor=str(platform.processor()),
        version=str(sys.version),
        cpu=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
    )
    try:
        import torch.cuda
    except ImportError:
        return cpu

    cpu["has_cuda"] = bool(torch.cuda.is_available())
    if cpu["has_cuda"]:
        cpu["capability"] = torch.cuda.get_device_capability(0)
        cpu["device_name"] = str(torch.cuda.get_device_name(0))
    return cpu


def _cmd_line(script_name: str, **kwargs: dict[str, Any]) -> list[str]:
    args = [sys.executable, "-m", script_name]
    for k, v in kwargs.items():
        args.append(f"--{k}")
        args.append(str(v))
    return args


def _extract_metrics(text: str) -> dict[str, str]:
    reg = re.compile(":(.*?),(.*.?);")
    res = reg.findall(text)
    if len(res) == 0:
        return {}
    return dict(res)


def _make_prefix(script_name: str, index: int) -> str:
    name = os.path.splitext(script_name)[0]
    return f"{name}_dort_c{index}_"


def run_benchmark(
    script_name: str,
    configs: list[dict[str, Any]],
    verbose: int = 0,
    stop_if_exception: bool = True,
    dump: bool = False,
) -> list[dict[str, Any]]:
    """
    Runs a script multiple times and extract information from the output
    following the pattern ``:<metric>,<value>;``.

    Args:
        script_name: python script to run
        configs: list of execution to do
        stop_if_exception: stop if one experiment failed, otherwise continue
        verbose: use tqdm to follow the progress
        dump: dump onnx file

    Returns:
        values
    """
    if verbose:
        from tqdm import tqdm

        loop = tqdm(configs)
    else:
        loop = configs

    data: list[dict[str, Any]] = []
    for i, config in enumerate(loop):
        cmd = _cmd_line(script_name, **config)

        if dump:
            os.environ["ONNXRT_DUMP_PATH"] = _make_prefix(script_name, i)
        else:
            os.environ["ONNXRT_DUMP_PATH"] = ""
        if verbose > 3:
            print(f"[run_benchmark] cmd={cmd if isinstance(cmd, str) else ' '.join(cmd)}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        out, err = res
        sout = out.decode("utf-8", errors="ignore")
        serr = err.decode("utf-8", errors="ignore")

        if "ONNXRuntimeError" in serr or "ONNXRuntimeError" in sout:
            if stop_if_exception:
                raise RuntimeError(
                    f"Unable to continue with config {config} due to the "
                    f"following error\n{serr}"
                    f"\n----OUTPUT--\n{sout}"
                )

        metrics = _extract_metrics(sout)
        if len(metrics) == 0:
            if stop_if_exception:
                raise BenchmarkError(
                    f"Unable (2) to continue with config {config}, no metric was "
                    f"collected.\n--ERROR--\n{serr}\n--OUTPUT--\n{sout}"
                )
            else:
                metrics = {}
        metrics.update(config)
        metrics["ERROR"] = serr
        metrics["OUTPUT"] = sout
        metrics["CMD"] = f"[{' '.join(cmd)}]"
        data.append(metrics)
        if verbose > 5:
            print("--------------- ERROR")
            print(serr)
        if verbose >= 10:
            print("--------------- OUTPUT")
            print(sout)

    return data


def measure_discrepancies(
    expected: list[tuple[Any, ...]],
    outputs: list[tuple[Any, ...]],
) -> tuple[float, float]:
    """
    Computes the discrepancies.

    Args:
        expected: list of outputs coming from a torch model
        outputs: list of outputs coming from an onnx model

    Returns:
        max absolute errors, max relative errors
    """

    def _flatten(outputs):
        flat = []
        for tensor in outputs:
            if isinstance(tensor, tuple):
                flat.extend(_flatten(tensor))
            else:
                flat.append(tensor)
        return tuple(flat)

    abs_errs = []
    rel_errs = []
    for torch_outputs_mixed_types, onnx_outputs in zip(expected, outputs):
        torch_outputs = _flatten(torch_outputs_mixed_types)
        assert len(torch_outputs) == len(
            onnx_outputs
        ), f"Length mismatch {len(torch_outputs)} != {len(onnx_outputs)}"
        for torch_tensor, onnx_tensor in zip(torch_outputs, onnx_outputs):
            assert (
                torch_tensor.dtype == onnx_tensor.dtype
            ), f"Type mismatch {torch_tensor.dtype} != {onnx_tensor.dtype}"
            assert (
                torch_tensor.shape == onnx_tensor.shape
            ), f"Type mismatch {torch_tensor.shape} != {onnx_tensor.shape}"
            diff = torch_tensor - onnx_tensor
            abs_err = float(diff.abs().max())
            rel_err = float((diff.abs() / torch_tensor).max())
            abs_errs.append(abs_err)
            rel_errs.append(rel_err)
    return max(abs_errs), max(rel_errs)


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
    stats: dict[str, Any] | None = None,
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
        optimization: optimization scenario, '/' separated values
        verbose: verbosity
        stats: if not None, populates this
            dictionary with statistics about time

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
            inputs,  # type: ignore[arg-type]
            filename,
            do_constant_folding=False,
            input_names=[f"input{i}" for i in range(len(inputs))],
            opset_version=target_opset,
            dynamic_axes=dynamic_shapes,
        )
    elif exporter == "dynamo":
        assert (
            dynamic_shapes is None
        ), f"dynamic_shapes={dynamic_shapes} is not implemented yet"
        with torch.no_grad():
            prog = torch.onnx.dynamo_export(model, *inputs)
        onnx.save(prog.model_proto, filename)
    else:
        raise ValueError(f"Unknown exporter {exporter!r}")

    if stats is not None:
        stats["export_time"] = time.perf_counter() - begin
        stats["filesize"] = os.stat(filename).st_size

    if verbose:
        print(f"[common_export] exporter done in {time.perf_counter() - begin}s")
        print(f"[common_export] size of the export: {os.stat(filename).st_size / 2**20} Mb")

    with open(filename, "rb") as f:
        onx = onnx.load(f)

    if optimization:
        if verbose:
            print(f"[common_export] start optimization with {optimization!r}")
        begin = time.perf_counter()
        optimized_model = optimize_model_proto(onx, optimization, verbose=verbose, stats=stats)
        end = time.perf_counter() - begin
        if stats is not None:
            stats["optimization_time"] = end
        if verbose:
            print(f"[common_export] optimization done in {end}")
            print(f"[common_export] saves the model in {filename!r}")
            begin = time.perf_counter()

        onnx.save(optimized_model, filename)
        if verbose:
            print(f"[common_export] done saving in {time.perf_counter() - begin}")

    return onx


def apply_rule_sets(
    model_proto: onnx.ModelProto,
    rule_sets: list[str],
    stats: dict[str, Any] | None = None,
    verbose: int = 0,
):
    """
    Applies set of patterns on a model to optimizes.

    Args:
        model_proto: model
        rule_sets: sets ot apply
        stats: add statistics if not empty
        verbose: verbosity

    Returns:
        optimized model
    """
    assert rule_sets, "No need to call apply_rule_sets for an empty set."
    if verbose:
        print(f"[apply_rule_sets] deserialize model before {rule_sets}")
    begin = time.perf_counter()
    ir_model = ir.serde.deserialize_model(model_proto)
    end = time.perf_counter() - begin
    if stats is not None:
        stats["deserialize_time"] = end
    if verbose:
        print(f"[apply_rule_sets] deserialize done in {end}")

    for rule_set_name in rule_sets:
        if verbose:
            print(f"[apply_rule_sets] applies {rule_set_name!r}")

        if rule_set_name == "llama0":
            rule_set = rules.llama_p0_rule_set()
        elif rule_set_name == "onnxruntime":
            rule_set = orp.RewriteRuleSet(ort_rules.ORT_PATTERN_REWRITE_RULES)
        else:
            raise ValueError(f"Unexpected rule_set name {rule_set_name!r}")

        begin = time.perf_counter()
        rule_set.apply_to_model(ir_model)
        remove_unused_nodes(ir_model)
        end = time.perf_counter() - begin
        if stats is not None:
            stats[f"opt_rule_{rule_set_name}_time"] = end
        if verbose:
            print(f"[apply_rule_sets] {rule_set_name} done in {end}")

    if verbose:
        print("[apply_rule_sets] serialize model")
    begin = time.perf_counter()
    rewritten_model = ir.serde.serialize_model(ir_model)
    end = time.perf_counter() - begin
    if stats is not None:
        stats["serialize_time"] = end
    if verbose:
        print(f"[apply_rule_sets] serialize done in {end}")

    if verbose:
        print("[apply_rule_sets] remove unused")
    begin = time.perf_counter()

    remove_unused_nodes(rewritten_model)

    end = time.perf_counter() - begin
    if stats is not None:
        stats["opt_remove_unused_time"] = end
    if verbose:
        print(f"[apply_rule_sets] remove unused done in {end}")

    return rewritten_model


def optimize_model_proto(
    model_proto: onnx.ModelProto,
    optimization: str | None = None,
    verbose: int = 0,
    stats: dict[str, Any] | None = None,
):
    """
    Optimizes a model given some scenarios.

    Args:
        model_proto: ModelProto
        optimization: '/' separated value
        verbose: verbosity
        stats: if not None, populates this dictionary with statistics

    Returns:
        optmized model
    """
    if not optimization:
        return model_proto

    known_rule_sets = {"llama0", "onnxruntime"}

    rule_sets: list[str] = []
    for value in optimization.split("/"):
        if value in known_rule_sets:
            rule_sets.append(value)
            continue
        if value not in known_rule_sets and rule_sets:
            model_proto = apply_rule_sets(model_proto, rule_sets, stats=stats, verbose=verbose)
            del rule_sets[:]
            continue

        if verbose:
            print(f"[optimize_model_proto] start {value}")

        n_nodes = len(model_proto.graph.node)
        n_functions = len(model_proto.functions)
        begin = time.perf_counter()

        if value == "optimize":
            model_proto = onnxscript.optimizer.optimize(
                model_proto,
                num_iterations=2,
                onnx_shape_inference=False,
            )

        elif value == "rewrite":
            model_proto = onnxscript.rewriter.rewrite(model_proto)

        elif value == "inline":
            model_proto = onnx.inliner.inline_local_functions(model_proto)

        else:
            raise AssertionError(
                f"Optimization step {value!r} is not implemented in {optimization!r}"
            )

        end = time.perf_counter() - begin
        delta = len(model_proto.graph.node) - n_nodes
        deltaf = len(model_proto.functions) - n_functions
        if stats:
            stats[f"opt_{value}_time"] = end
            stats[f"opt_{value}_dnodes"] = delta
            stats[f"opt_{value}_dfunctions"] = deltaf
        if verbose:
            print(
                f"[optimize_model_proto] {value} done in {end} "
                f"with +/- {delta} nodes, +/- {deltaf} functions"
            )
    if rule_sets:
        model_proto = apply_rule_sets(model_proto, rule_sets, stats=stats, verbose=verbose)

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

    stats: dict[str, Any] = {}
    iterations: list[float] = []
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


class WrapInferenceSessionForTorch:
    def __init__(self, sess: Any):
        # onnxruntime is importing when needed as it takes a couple of seconds if it contains CUDA EP.
        import onnxruntime
        import torch
        from onnxruntime.capi import _pybind_state as ORTC  # noqa: N812

        self.sess = sess
        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [i.name for i in sess.get_outputs()]
        self.bind = onnxruntime.SessionIOBinding(sess._sess)
        self.OrtValue = ORTC.OrtValue
        self.ORTC = ORTC
        self.torch = torch
        self.run_options = onnxruntime.RunOptions()

        self.TORCH_DTYPE_TO_NUMPY_DTYPE = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bool: np.bool_,
        }

        DEVICES = {
            -1: ORTC.OrtDevice(ORTC.OrtDevice.cpu(), ORTC.OrtDevice.default_memory(), 0)
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                DEVICES[i] = ORTC.OrtDevice(
                    ORTC.OrtDevice.cuda(), ORTC.OrtDevice.default_memory(), i
                )

        self.DEVICES = DEVICES

    def _get_ortvalues_from_torch_tensors(
        self,
        tensors: tuple[Any, ...],  # tuple["torch.Tensor", ...],
        n_outputs: int,
    ) -> tuple[Any, Any]:  # tuple[tuple["torch.Tensor", ...], tuple["OrtDevice", ...]]:
        ortvalues = self.ORTC.OrtValueVector()
        ortvalues.reserve(len(tensors))
        dtypes = []
        shapes = []
        data_ptrs = []
        devices = []

        max_device = -1
        assert isinstance(max_device, int), f"unexpected type for device={max_device!r}"
        assert tensors is not None, "tensors cannot be None"
        new_tensors = []
        for tensor in tensors:
            assert isinstance(tensor, self.torch.Tensor), f"Unexpected type {type(tensor)}"
            dtypes.append(self.TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])
            shapes.append(tensor.size())
            data_ptrs.append(tensor.data_ptr())
            d = tensor.get_device()
            devices.append(self.DEVICES[d])
            new_tensors.append(tensor)
            max_device = max(max_device, tensor.get_device())

        ortvalues.push_back_batch(new_tensors, data_ptrs, dtypes, shapes, devices)
        output_devices = []
        for _ in range(n_outputs):
            dev = self.DEVICES[max_device]
            output_devices.append(dev)

        return ortvalues, output_devices

    def _ortvalues_to_torch_tensor(
        self,
        ortvalues: Any,  #  "onnxruntime.OrtValueVector",
    ) -> tuple[Any, ...]:  # tuple["torch.Tensor", ...]:
        if len(ortvalues) == 0:
            return tuple()

        from torch._C import _from_dlpack

        if all(map(lambda i: ortvalues[i].has_value(), range(len(ortvalues)))):  # noqa: C417
            res = ortvalues.to_dlpacks(_from_dlpack)
        else:
            res = []
            for i in range(len(ortvalues)):
                res.append(
                    _from_dlpack(ortvalues[i].to_dlpack())
                    if ortvalues[i].has_value()
                    else None
                )
        return tuple(res)

    def run(self, output_names, feeds):
        inputs = [feeds[i] for i in self.input_names]
        return self.run_dlpack(*inputs, output_names=output_names)

    def run_dlpack(self, *inputs, output_names=None):
        if output_names is None:
            output_names = self.output_names
        ortvalues, output_devices = self._get_ortvalues_from_torch_tensors(
            inputs, len(output_names)
        )

        ort_outputs = self.ORTC.OrtValueVector()
        self.sess.run_with_ortvaluevector(
            self.run_options,
            self.input_names,
            ortvalues,
            output_names,
            ort_outputs,
            output_devices,
        )
        pth_outputs = self._ortvalues_to_torch_tensor(ort_outputs)
        return pth_outputs


def run_onnx_inference(
    model: onnx.ModelProto,
    example_inputs: Sequence[Any],
    warmup: int = 5,
    repeat: int = 5,
    verbose: int = 0,
    ort_optimize: bool = True,
    torch_model: Any | None = None,
) -> dict[str, Any]:
    """
    Runs multiple times the same inference with onnxruntime.

    Args:
        model: torch model to run
        example_inputs: dummy inputs
        warmup: number of iterations to warmup
        repeat: number of iterations to repeat
        verbose: verbosity
        ort_optimize: enable, disable onnxruntime optimizations
        torch_model: if not empty, measure the discrepancies

    Returns:
        statistcs
    """
    stats: dict[str, Any] = {}
    device = example_inputs[0][0].get_device()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device >= 0
        else ["CPUExecutionProvider"]
    )
    stats["providers"] = ",".join(providers)
    if verbose:
        print(f"[run_inference] create session with providers {providers!r}")

    begin = time.perf_counter()
    # onnxruntime is importing when needed as it takes a couple of seconds if it contains CUDA EP.
    import onnxruntime

    so = onnxruntime.SessionOptions()
    if ort_optimize:
        so.add_session_config_entry("session.disable_aot_function_inlining", "0")
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        so.add_session_config_entry("session.disable_aot_function_inlining", "1")
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = onnxruntime.InferenceSession(model.SerializeToString(), so, providers)
    wrapped_session = WrapInferenceSessionForTorch(sess)

    end = time.perf_counter() - begin
    stats["ort_session_create_time"] = end
    if verbose:
        print(f"[run_inference] created session in {end}")
        print(f"[run_inference] start {warmup} warmup iterations")

    if torch_model:
        expected = [
            torch_model(*example_inputs[i % len(example_inputs)]) for i in range(warmup)
        ]

    got = []
    iterations = []
    begin = time.perf_counter()
    for i in range(warmup):
        t0 = time.perf_counter()
        got.append(wrapped_session.run_dlpack(*example_inputs[i % len(example_inputs)]))
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["warmup"] = warmup
    stats["warmup_time"] = end / warmup
    stats["warmup_iter"] = iterations
    if torch_model:
        abs_err, rel_err = measure_discrepancies(expected, got)
        stats["discrepancies_abs"] = abs_err
        stats["discrepancies_rel"] = rel_err

    if verbose:
        print(f"[run_inference] warmup done in {time.perf_counter() - begin}")
        print(f"[run_inference] start {repeat} iterations")

    iterations = []
    begin = time.perf_counter()
    for i in range(repeat):
        t0 = time.perf_counter()
        wrapped_session.run_dlpack(*example_inputs[i % len(example_inputs)])
        iterations.append(time.perf_counter() - t0)
    end = time.perf_counter() - begin
    stats["repeat"] = repeat
    stats["repeat_time"] = end / repeat
    stats["repeat_iter"] = iterations

    if verbose:
        print(f"[run_inference] measure done in {time.perf_counter() - begin}")

    return stats


def multi_run(kwargs: dict[str, Any]) -> bool:
    """Checks if multiple values were sent for one argument."""
    return any(isinstance(v, str) and "," in v for v in kwargs.values())


def make_configs(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Creates all the configurations based on the command line arguments."""
    print(kwargs)
    args = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            args.append([(k, s) for s in v.split(",")])
        else:
            args.append([(k, v)])
    configs = list(itertools.product(*args))
    return [dict(c) for c in configs]


def make_dataframe_from_benchmark_data(data: list[dict]) -> Any:
    """Creates a dataframe from the received data."""
    import pandas

    return pandas.DataFrame(data)
