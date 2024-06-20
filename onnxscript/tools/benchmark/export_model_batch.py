# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import pprint
import textwrap
from typing import Any

import onnxscript.tools.benchmark


def main(args: list[str] | None = None):
    kwargs: dict[str, Any] = onnxscript.tools.benchmark.get_parsed_args(
        "export_model",
        description=textwrap.dedent(
            """Measures the inference time for a particular model.
            It runs export_model to compare several optimization settings.

            Example::

                python -m onnxscript.tools.benchmark.export_model_batch --model phi --device cuda --config medium --num_hidden_layers=1 --dtype=float32 --dynamic=0 --verbose=1
            """
        ),
        repeat=(10, "number of inferences to measure"),
        warmup=(5, "number of inferences to warm"),
        model=("phi", "model to measure, llama, mistral, phi, ..."),
        device=("cpu", "'cpu' or 'cuda'"),
        target_opset=(18, "opset to convert into, use with backend=custom"),
        config=("small", "default, medium, or small to test"),
        verbose=(0, "verbosity"),
        dtype=("default", "cast the model and the inputs into this type"),
        dynamic=(0, "use dynamic shapes"),
        num_hidden_layers=(1, "number of hidden layers"),
        with_mask=(1, "with or without mask, dynamo may fail with a mask"),
        implementation=("eager", "eager or sdpa"),
        new_args=args,
    )

    print("-------------------")
    print("[export_model]")
    pprint.pprint(kwargs)
    print("-------------------")

    import pandas

    try:
        import openpyxl
    except ImportError:
        openpyxl = None

    from onnxscript.tools.benchmark.benchmark_helpers import (
        BenchmarkError,
        run_benchmark,
    )

    script_name = "onnxscript.tools.benchmark.export_model"

    configs: list[dict[str, Any]] = [
        dict(exporter="eager"),
        dict(ort_optimize=1, exporter="script"),
        dict(ort_optimize=1, optimization="optimize/rewrite/inline", exporter="script"),
        dict(ort_optimize=0, optimization="optimize/rewrite/inline", exporter="script"),
        dict(ort_optimize=1, optimization="", exporter="dynamo"),
        dict(ort_optimize=1, optimization="optimize/rewrite/inline", exporter="dynamo"),
        dict(ort_optimize=0, optimization="optimize/rewrite/inline", exporter="dynamo"),
    ]
    common_kwargs: dict[str, Any] = kwargs.copy()
    common_kwargs["verbose"] = max(common_kwargs["verbose"] - 1, 0)
    for c in configs:
        c.update(common_kwargs)

    if kwargs["verbose"]:
        for i, cf in enumerate(configs):
            print(f"[export_common_batch] config {i+1}: {cf}")

    ################################
    # Running configuration.

    try:
        data = run_benchmark(
            script_name,
            configs,
            verbose=kwargs["verbose"],
            stop_if_exception=False,
        )
        data_collected = True
    except BenchmarkError as e:
        if kwargs["verbose"]:
            print(e)
        data_collected = False

    prefix = "_".join(
        [
            "emb_",
            kwargs["model"],
            "dynamic" if kwargs["dynamic"] else "static",
            kwargs["dtype"].replace("float", "fp"),
            kwargs["device"],
            kwargs["config"],
            f"h{kwargs['num_hidden_layers']}",
        ],
    )

    if data_collected:
        df = pandas.DataFrame(data)
        df = df.drop(["OUTPUT", "ERROR"], axis=1)
        df["repeat_time"] = df["repeat_time"].astype(float)
        df_eager = df[(df["implementation"] == "eager") & (df["exporter"] == "eager")][
            "repeat_time"
        ].dropna()
        if df_eager.shape[0] > 0:
            min_eager = df_eager.min()
            df["increase"] = df["repeat_time"] / min_eager - 1
        filename = f"{prefix}_with_cmd.csv"
        df.to_csv(filename, index=False)

        df = df.drop(["CMD"], axis=1)
        filename = f"{prefix}.csv"
        df.to_csv(filename, index=False)
        df = pandas.read_csv(filename)  # to cast type
        print(df)

        # summary
        cs = [
            c
            for c in ["exporter", "optimization", "warmup_time", "repeat_time", "increase"]
            if c in df.columns
        ]
        dfs = df[cs]
        if openpyxl:
            filename = f"{prefix}_summary.xlsx"
            dfs.to_excel(filename, index=False)
        filename = f"{prefix}_summary.csv"
        dfs.to_csv(filename, index=False)
        print(dfs)

    ########################
    # First lines.

    print(df.head(2).T)


if __name__ == "__main__":
    main()
