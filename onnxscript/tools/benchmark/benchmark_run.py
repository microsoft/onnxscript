# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=consider-using-with,import-outside-toplevel
from __future__ import annotations

import multiprocessing
import os
import platform
import re
import subprocess
import sys


class BenchmarkError(RuntimeError):
    pass


def get_machine() -> dict[str, str | int | float | tuple[int, int]]:
    """Returns the machine specification."""
    config: dict[str, str | int | float | tuple[int, int]] = dict(
        machine=str(platform.machine()),
        processor=str(platform.processor()),
        version=str(sys.version),
        config=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
    )
    try:
        import torch.cuda
    except ImportError:
        return config

    config["has_cuda"] = bool(torch.cuda.is_available())
    if config["has_cuda"]:
        config["capability"] = torch.cuda.get_device_capability(0)
        config["device_name"] = str(torch.cuda.get_device_name(0))
    return config


def _cmd_line(script_name: str, **kwargs: dict[str, str | int | float]) -> list[str]:
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
    configs: list[dict[str, str | int | float]],
    verbose: int = 0,
    stop_if_exception: bool = True,
    dort_dump: bool = False,
) -> list[dict[str, str | int | float | tuple[int, int]]]:
    """
    Runs a script multiple times and extract information from the output
    following the pattern ``:<metric>,<value>;``.

    :param script_name: python script to run
    :param configs: list of execution to do
    :param stop_if_exception: stop if one experiment failed, otherwise continue
    :param verbose: use tqdm to follow the progress
    :param dort_dump: dump onnx file if dort is used
    :return: values
    """
    if verbose:
        try:
            from tqdm import tqdm

            loop = tqdm(configs)
        except ImportError:
            loop = configs
    else:
        loop = configs

    data: list[dict[str, str | int | float | tuple[int, int]]] = []
    for i, config in enumerate(loop):
        cmd = _cmd_line(script_name, **config)

        if dort_dump:
            os.environ["ONNXRT_DUMP_PATH"] = _make_prefix(script_name, i)
        else:
            os.environ["ONNXRT_DUMP_PATH"] = ""
        if verbose > 3:
            print(f"[run_benchmark] cmd={cmd if isinstance(cmd, str) else ' '.join(cmd)}")

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            res = p.communicate(timeout=30)
            out, err = res
            serr = err.decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired as e:
            p.kill()
            res = p.communicate()
            out, err = res
            serr = f"{e}\n:timeout,1;{err.decode('utf-8', errors='ignore')}"
        sout = out.decode("utf-8", errors="ignore")

        if "ONNXRuntimeError" in serr or "ONNXRuntimeError" in sout:
            if stop_if_exception:  # pylint: disable=no-else-raise
                raise RuntimeError(
                    f"Unable to continue with config {config} due to the "
                    f"following error\n{serr}"
                    f"\n----OUTPUT--\n{sout}"
                )

        metrics = _extract_metrics(sout)
        if len(metrics) == 0:
            if stop_if_exception:  # pylint: disable=no-else-raise
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
        data.append(metrics)  # type: ignore[arg-type]
        if verbose > 5:
            print("--------------- ERROR")
            print(serr)
        if verbose >= 10:
            print("--------------- OUTPUT")
            print(sout)

    return data
