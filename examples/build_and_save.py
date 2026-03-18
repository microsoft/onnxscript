#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build and save — the simplest possible example.

Demonstrates the two-line workflow: ``build()`` a model package from a
HuggingFace model ID, then ``save()`` it to disk.

Usage::

    # Build a causal LM
    python examples/build_and_save.py --model Qwen/Qwen2.5-0.5B output/qwen2.5/

    # Build without downloading weights (graph skeleton only)
    python examples/build_and_save.py --model meta-llama/Llama-3.2-1B output/llama/ --no-weights

    # Build with a specific dtype
    python examples/build_and_save.py --model meta-llama/Llama-3.2-1B output/llama/ --dtype f16

    # Build a diffusers pipeline (produces subfolders per component)
    python examples/build_and_save.py --model Qwen/Qwen-Image-2512 output/qwen-image/

    # Build an encoder-decoder model (produces encoder/ and decoder/ subfolders)
    python examples/build_and_save.py --model openai/whisper-tiny output/whisper/
"""

from __future__ import annotations

import argparse

from mobius import build


def main():
    parser = argparse.ArgumentParser(
        description="Build an ONNX model from a HuggingFace model ID and save it.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-1B').",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save the ONNX model package.",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Build the graph skeleton only (no weight download).",
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16"],
        default=None,
        help="Override the model dtype.",
    )
    args = parser.parse_args()

    print(f"Building {args.model!r} ...")
    pkg = build(
        args.model,
        load_weights=not args.no_weights,
        dtype=args.dtype,
    )

    print(f"Package components: {list(pkg.keys())}")
    pkg.save(
        args.output_dir,
        check_weights=not args.no_weights,
    )
    print("Done.")


if __name__ == "__main__":
    main()
