# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for mobius."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import TYPE_CHECKING

import tqdm

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch

from mobius._builder import (
    DTYPE_MAP,
    build,
    build_from_module,
    resolve_dtype,
)
from mobius._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
)
from mobius._registry import registry

logger = logging.getLogger(__name__)


def _parse_size(size_str: str) -> int:
    """Parse a human-readable size string (e.g. '5GB') to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


def _load_weights_from_dir(model_dir: str) -> dict[str, torch.Tensor]:
    """Load safetensors weights from a local model directory."""
    import safetensors.torch

    model_dir = os.path.abspath(model_dir)
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        all_files = sorted(set(index["weight_map"].values()))
        paths = [os.path.join(model_dir, f) for f in all_files]
    else:
        paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    if not paths:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc="Loading weights"):
        state_dict.update(safetensors.torch.load_file(path))
    return state_dict


def _apply_optimize(model: ir.Model, optimize: str | None) -> None:
    """Apply rewrite rules if --optimize is specified."""
    if not optimize:
        return

    from onnxscript.rewriter import rewrite

    from mobius.rewrite_rules import (
        bias_gelu_rules,
        fused_matmul_rules,
        group_query_attention_rules,
        packed_attention_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
    )

    rule_map = {
        "bias_gelu": bias_gelu_rules,
        # group_query_attention (incl. QKV packing) must run before
        # fused_matmul so that projections are still plain MatMul nodes
        # when the packing pattern matches.
        "group_query_attention": group_query_attention_rules,
        "fused_matmul": fused_matmul_rules,
        "packed_attention": packed_attention_rules,
        "skip_layer_norm": skip_layer_norm_rules,
        "skip_norm": skip_norm_rules,
    }

    if optimize == "all":
        rule_names = list(rule_map)
    else:
        rule_names = [r.strip() for r in optimize.split(",")]
        for name in rule_names:
            if name not in rule_map:
                raise ValueError(
                    f"Unknown rewrite rule '{name}'. Available: {sorted(rule_map)}"
                )

    for name in rule_names:
        rules = rule_map[name]()
        rewrite(model, pattern_rewrite_rules=rules)
        print(f"Applied rewrite rule: {name}")


def _cmd_build(args: argparse.Namespace) -> None:
    """Execute the 'build' subcommand."""
    import dataclasses

    from mobius._diffusers_builder import (
        _load_diffusers_pipeline_index,
        build_diffusers_pipeline,
    )
    from mobius.tasks import CausalLMTask, ModelTask

    # Validate --max-seq-len requires --static-cache
    if args.max_seq_len is not None and not args.static_cache:
        raise SystemExit("Error: --max-seq-len can only be used with --static-cache.")

    # Validate --max-seq-len is positive
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise SystemExit("Error: --max-seq-len must be a positive integer.")

    # Validate --static-cache + --task compatibility
    if args.static_cache and args.task is not None:
        raise SystemExit(
            "Error: --static-cache cannot be combined with --task. "
            "Remove --task to use --static-cache."
        )

    load_weights = not args.no_weights
    task: str | ModelTask | None = args.task
    if args.static_cache:
        task = CausalLMTask(static_cache=True, max_seq_len=args.max_seq_len)
    trust_remote_code = args.trust_remote_code
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dtype_override = resolve_dtype(args.dtype)
    optimize = args.optimize
    component_filter = args.component

    # Auto-detect diffusers pipelines
    if args.model and not args.config:
        pipeline_index = _load_diffusers_pipeline_index(args.model)
        if pipeline_index is not None:
            print(
                f"Detected diffusers pipeline: {pipeline_index.get('_class_name', 'Unknown')}"
            )
            pkg = build_diffusers_pipeline(
                args.model,
                dtype=dtype_override,
                load_weights=load_weights,
            )
            _save_package(pkg, output_dir, args, optimize, component_filter)
            return

    # Build from HuggingFace model ID or local config
    if args.config:
        import transformers

        config_path = args.config
        hf_config = transformers.AutoConfig.from_pretrained(
            config_path, trust_remote_code=trust_remote_code
        )
        model_type = hf_config.model_type
        parent_config = hf_config
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        config = _config_from_hf(hf_config, parent_config=parent_config)
        if dtype_override is not None:
            config = dataclasses.replace(config, dtype=dtype_override)
        if task is None:
            task = _default_task_for_model(model_type)
        module_class = registry.get(model_type)
        model_module = module_class(config)
        pkg = build_from_module(model_module, config, task=task)
        for name, model in pkg.items():
            model.graph.name = f"{config_path}/{name}"
        if load_weights:
            state_dict = _load_weights_from_dir(config_path)
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            pkg.apply_weights(state_dict)
    else:
        pkg = build(
            args.model,
            task=task,
            dtype=dtype_override,
            load_weights=load_weights,
            trust_remote_code=trust_remote_code,
        )

    _save_package(pkg, output_dir, args, optimize, component_filter)


def _save_package(
    pkg, output_dir: str, args, optimize: str | None, component_filter: str | None
) -> None:
    """Save a ModelPackage to disk, applying optimizations."""
    components = (lambda name: name == component_filter) if component_filter else None
    for name, model in pkg.items():
        if components is not None and not components(name):
            continue
        _apply_optimize(model, optimize)

    max_shard_size_bytes = _parse_size(args.max_shard_size) if args.max_shard_size else None
    pkg.save(
        output_dir,
        external_data=args.external_data,
        max_shard_size_bytes=max_shard_size_bytes,
        components=components,
        check_weights=not args.no_weights,
    )
    selected = [name for name in pkg if components is None or components(name)]
    use_subfolders = len(selected) > 1
    for name in selected:
        if use_subfolders:
            path = os.path.join(output_dir, name, "model.onnx")
        else:
            path = os.path.join(output_dir, "model.onnx")
        print(f"Saved {name} to {path}")


def _cmd_list(args: argparse.Namespace) -> None:
    """Execute the 'list' subcommand."""
    from mobius.tasks import TASK_REGISTRY

    resource = args.resource
    if resource == "models":
        architectures = registry.architectures()
        print(f"Supported model architectures ({len(architectures)}):\n")
        for arch in architectures:
            module_class = registry.get(arch)
            task = getattr(module_class, "default_task", "text-generation")
            category = getattr(module_class, "category", "")
            print(f"  {arch:<30} task={task:<25} category={category}")
    elif resource == "tasks":
        print(f"Available tasks ({len(TASK_REGISTRY)}):\n")
        for name in sorted(TASK_REGISTRY):
            cls = TASK_REGISTRY[name]
            print(f"  {name:<35} {cls.__name__}")
    elif resource == "dtypes":
        seen: set[str] = set()
        print("Available dtypes:\n")
        for _name, dt in sorted(DTYPE_MAP.items()):
            if dt.name not in seen:
                aliases = [k for k, v in DTYPE_MAP.items() if v == dt]
                print(f"  {' | '.join(aliases):<25} → {dt.name}")
                seen.add(dt.name)
    else:
        print(f"Unknown resource '{resource}'. Use: models, tasks, dtypes")


def _cmd_build_gguf(args: argparse.Namespace) -> None:
    """Execute the 'build-gguf' subcommand."""
    try:
        from mobius.integrations.gguf import build_from_gguf
    except ImportError:
        print(
            "GGUF support requires the gguf package. Install with: pip install mobius-ai[gguf]"
        )
        raise SystemExit(1)

    if args.keep_quantized:
        print("Quantized mode: preserving GGUF quantization as MatMulNBits...")

    gguf_path = args.gguf_path
    output_dir = args.output or os.path.splitext(gguf_path)[0] + "_onnx"
    os.makedirs(output_dir, exist_ok=True)

    pkg = build_from_gguf(
        gguf_path,
        dtype=args.dtype,
        keep_quantized=args.keep_quantized,
    )

    pkg.save(
        output_dir,
        external_data=args.external_data,
    )
    for name in pkg:
        use_subfolders = len(pkg) > 1
        if use_subfolders:
            path = os.path.join(output_dir, name, "model.onnx")
        else:
            path = os.path.join(output_dir, "model.onnx")
        print(f"Saved {name} to {path}")


def _cmd_info(args: argparse.Namespace) -> None:
    """Execute the 'info' subcommand."""
    from mobius._diffusers_builder import (
        _DIFFUSERS_CLASS_MAP,
        _init_diffusers_class_map,
        _load_diffusers_pipeline_index,
    )

    model_id = args.model_id

    # Check diffusers first
    pipeline_index = _load_diffusers_pipeline_index(model_id)
    if pipeline_index is not None:
        pipeline_class = pipeline_index.get("_class_name", "Unknown")
        print(f"Model:    {model_id}")
        print(f"Type:     Diffusers pipeline ({pipeline_class})")
        print("Components:")
        for comp_name, info in pipeline_index.items():
            if comp_name.startswith("_") or not isinstance(info, list):
                continue
            library, class_name = info
            _init_diffusers_class_map()
            supported = "✓" if class_name in _DIFFUSERS_CLASS_MAP else "✗"
            print(f"  {supported} {comp_name:<20} {class_name} ({library})")
        return

    # Try transformers
    import transformers

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code
        )
    except (OSError, ValueError) as e:
        logger.debug("Failed to load config for '%s': %s", model_id, e)
        print(f"Error loading config for '{model_id}': {e}")
        return

    model_type = hf_config.model_type
    in_registry = model_type in registry

    print(f"Model:         {model_id}")
    print(f"Model type:    {model_type}")
    print(f"Supported:     {'✓' if in_registry else '✗ (not registered)'}")

    if in_registry:
        module_class = registry.get(model_type)
        task = getattr(module_class, "default_task", "text-generation")
        category = getattr(module_class, "category", "")
        print(f"Module class:  {module_class.__name__}")
        print(f"Default task:  {task}")
        print(f"Category:      {category}")

    # Show key config values
    print("Config:")
    for field in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "intermediate_size",
        "torch_dtype",
    ]:
        val = getattr(hf_config, field, None)
        if val is not None:
            print(f"  {field}: {val}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mobius",
        description="Build ONNX models for GenAI from HuggingFace model architectures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    build_parser = subparsers.add_parser("build", help="Build an ONNX model.")
    source_group = build_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. 'meta-llama/Llama-3-8B').",
    )
    source_group.add_argument(
        "--config",
        metavar="CONFIG_PATH",
        help="Path to a local model directory containing config.json (and optionally safetensors weights).",
    )
    build_parser.add_argument(
        "output_dir",
        help="Output directory for the ONNX model.",
    )
    build_parser.add_argument(
        "--task",
        default=None,
        help="Model task (auto-detected if not specified). Use 'mobius list tasks' to see available tasks.",
    )
    build_parser.add_argument(
        "--external-data",
        choices=["onnx", "safetensors"],
        default="onnx",
        help="External data format (default: onnx).",
    )
    build_parser.add_argument(
        "--max-shard-size",
        metavar="SIZE",
        default=None,
        help="Max shard size for safetensors (e.g. '5GB'). Only used with --external-data safetensors.",
    )
    build_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Do not include weights in the output ONNX model.",
    )
    build_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the HuggingFace model config.",
    )
    build_parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Target dtype for model weights (default: f32). Weights are cast at save time.",
    )
    build_parser.add_argument(
        "--optimize",
        nargs="?",
        const="all",
        default=None,
        metavar="RULES",
        help=(
            "Apply rewrite rules after building. "
            "Use without value for all rules, or specify comma-separated rule names "
            "(e.g. --optimize=group_query_attention,skip_norm). "
            "Available: group_query_attention, packed_attention, skip_norm."
        ),
    )
    build_parser.add_argument(
        "--component",
        default=None,
        metavar="NAME",
        help="Build only this component from a diffusers pipeline (e.g. --component vae_decoder).",
    )
    build_parser.add_argument(
        "--static-cache",
        action="store_true",
        help="Use static KV cache (pre-allocated buffers with TensorScatter). "
        "Requires models using DecoderLayer or MoEDecoderLayer.",
    )
    build_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum sequence length for static cache buffers. "
        "Only used with --static-cache. Defaults to max_position_embeddings from config.",
    )
    build_parser.set_defaults(func=_cmd_build)

    # --- build-gguf ---
    gguf_parser = subparsers.add_parser(
        "build-gguf", help="Build ONNX model from a GGUF file."
    )
    gguf_parser.add_argument(
        "gguf_path",
        help="Path to a .gguf model file.",
    )
    gguf_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="DIR",
        help="Output directory (default: <gguf_stem>_onnx/).",
    )
    gguf_parser.add_argument(
        "--keep-quantized",
        action="store_true",
        help="Preserve quantization via MatMulNBits (Q4_0/Q4_1/Q8_0).",
    )
    gguf_parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Target dtype for model weights.",
    )
    gguf_parser.add_argument(
        "--external-data",
        choices=["onnx", "safetensors"],
        default="onnx",
        help="External data format (default: onnx).",
    )
    gguf_parser.set_defaults(func=_cmd_build_gguf)

    # --- list ---
    list_parser = subparsers.add_parser(
        "list", help="List supported models, tasks, or dtypes."
    )
    list_parser.add_argument(
        "resource",
        choices=["models", "tasks", "dtypes"],
        help="What to list.",
    )
    list_parser.set_defaults(func=_cmd_list)

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show information about a model.")
    info_parser.add_argument(
        "model_id",
        help="HuggingFace model ID to inspect.",
    )
    info_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the HuggingFace model config.",
    )
    info_parser.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
