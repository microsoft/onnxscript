#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion model example — build and save ONNX image generation models.

Demonstrates building diffusion pipeline components (transformer, VAE, etc.)
from a HuggingFace diffusers model using ``build_diffusers_pipeline``, then
optionally running a VAE decode to verify the model works.

Supported architectures:
    - Flux (``FluxTransformer2DModel``)
    - Stable Diffusion 3 (``SD3Transformer2DModel``)
    - PixArt / DiT (``DiTTransformer2DModel``, ``PixArtTransformer2DModel``)
    - QwenImage (``QwenImageTransformer2DModel``)
    - AutoencoderKL (VAE for all above)

Usage::

    # Build and save a Flux model:
    python examples/diffusion.py --model black-forest-labs/FLUX.1-schnell output/flux/

    # Build an SD3 model:
    python examples/diffusion.py --model stabilityai/stable-diffusion-3.5-large-turbo output/sd3/

    # Build without weights (graph skeleton only):
    python examples/diffusion.py --model black-forest-labs/FLUX.1-schnell output/flux/ --no-weights

    # Build with f16 precision:
    python examples/diffusion.py --model black-forest-labs/FLUX.1-schnell output/flux/ --dtype f16

    # Run a quick VAE decode test after building:
    python examples/diffusion.py --model black-forest-labs/FLUX.1-schnell output/flux/ --test-vae

Note:
    This builds the neural-network components (transformer, VAE) as ONNX
    models. A full image generation pipeline also requires a text encoder
    and a noise scheduler, which are not built by this package. Use the
    built ONNX models with your own scheduler loop or integrate with
    frameworks like ONNX Runtime GenAI.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Build diffusion ONNX models from a HuggingFace diffusers pipeline.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace diffusers model ID (e.g. 'black-forest-labs/FLUX.1-schnell').",
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
    parser.add_argument(
        "--test-vae",
        action="store_true",
        help="Run a quick VAE decode test after building (requires onnxruntime).",
    )
    args = parser.parse_args()

    # Lazy import to give a clear error if the package is not installed
    try:
        from mobius import build_diffusers_pipeline
    except ImportError:
        print(
            "Error: mobius is not installed.\n"
            "Install with: pip install mobius-ai[transformers]",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Build the diffusion pipeline components
    # -----------------------------------------------------------------------
    print(f"Building diffusers pipeline {args.model!r} ...")
    pkg = build_diffusers_pipeline(
        args.model,
        load_weights=not args.no_weights,
        dtype=args.dtype,
    )

    print(f"\nComponents built: {list(pkg.keys())}")
    for name, model in pkg.items():
        n_params = sum(1 for name in model.graph.initializers if not name.startswith("const_"))
        print(f"  {name}: {n_params} parameters, {model.graph.num_nodes} ops")

    # -----------------------------------------------------------------------
    # Step 2: Save the ONNX models
    # -----------------------------------------------------------------------
    print(f"\nSaving to {args.output_dir!r} ...")
    pkg.save(
        args.output_dir,
        check_weights=not args.no_weights,
    )
    print("Done.")

    # -----------------------------------------------------------------------
    # Step 3 (optional): Quick VAE decode test
    # -----------------------------------------------------------------------
    if args.test_vae:
        _test_vae_decode(pkg, args.output_dir)


def _test_vae_decode(pkg, output_dir: str):
    """Run a quick VAE decode to verify the model works."""
    # Find a VAE component
    vae_key = None
    for key in pkg:
        if "vae" in key.lower():
            vae_key = key
            break

    if vae_key is None:
        print("\nNo VAE component found — skipping decode test.")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "\nonnxruntime not installed — skipping VAE decode test.\n"
            "Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        return

    import os

    import onnx_ir

    print(f"\nRunning VAE decode test with '{vae_key}' ...")

    # Save and load with ORT
    vae_model = pkg[vae_key]
    vae_path = os.path.join(output_dir, vae_key, "model.onnx")
    if not os.path.exists(vae_path):
        # Single-component package saves directly
        vae_path = os.path.join(output_dir, "model.onnx")

    if not os.path.exists(vae_path):
        # Save just the VAE for testing in a temp directory
        import tempfile

        _tmpdir = tempfile.mkdtemp(prefix="onnx_genai_vae_test_")
        vae_path = os.path.join(_tmpdir, "model.onnx")
        onnx_ir.save(vae_model, vae_path)
    else:
        _tmpdir = None

    try:
        sess = ort.InferenceSession(vae_path, providers=["CPUExecutionProvider"])

        # Create random latent input matching the first input shape
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        shape = []
        for dim in input_info.shape:
            shape.append(dim if isinstance(dim, int) else 1)

        rng = np.random.default_rng(42)
        latent = rng.standard_normal(shape).astype(np.float32)

        print(f"  Input '{input_name}': shape={shape}")
        outputs = sess.run(None, {input_name: latent})
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        print("VAE decode test passed.")
    finally:
        if _tmpdir is not None:
            import shutil

            shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
