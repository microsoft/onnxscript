#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Architecture diff CLI.

Detects which ONNX model architectures are affected by changes between
two git refs, builds tiny graphs for each, and produces a Markdown diff
report.

Usage::

    python scripts/arch_diff.py \
        --base-ref HEAD~1 --head-ref HEAD --output arch_diff.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Ensure the project root is on sys.path so imports work when run as
# a standalone script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "tests"))

_GITHUB_REPO_URL = "https://github.com/onnxruntime/mobius"


# ------------------------------------------------------------------
# Model build configs — mirrors the benchmark / test infrastructure
# ------------------------------------------------------------------
# Each entry: (model_type, config_overrides, task_name, build_kind)
#   build_kind is "standard", "whisper", "mamba", or "qwen3_5_vl".
_DIFF_MODELS: list[tuple[str, dict, str, str]] = [
    ("llama", {}, "text-generation", "standard"),
    ("llama", {}, "static-cache", "standard"),
    ("qwen2", {}, "text-generation", "standard"),
    ("qwen2", {}, "static-cache", "standard"),
    ("qwen", {}, "text-generation", "standard"),
    ("qwen", {}, "static-cache", "standard"),
    ("qwen3", {"attn_qk_norm": True}, "text-generation", "standard"),
    ("qwen3", {"attn_qk_norm": True}, "static-cache", "standard"),
    (
        "qwen3_5_text",
        {
            "partial_rotary_factor": 0.5,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        "hybrid-text-generation",
        "standard",
    ),
    (
        "qwen3_5_moe",
        {
            "hidden_act": "silu",
            "layer_types": ["linear_attention", "full_attention"],
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        "hybrid-text-generation",
        "standard",
    ),
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_hidden_layers": 4,
            "partial_rotary_factor": 0.25,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "attn_qk_norm": True,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        "hybrid-text-generation",
        "standard",
    ),
    (
        "qwen2_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 64,
            "attn_qkv_bias": True,
        },
        "text-generation",
        "standard",
    ),
    (
        "qwen2_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 64,
            "attn_qkv_bias": True,
        },
        "static-cache",
        "standard",
    ),
    (
        "qwen3_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qk_norm": True,
        },
        "text-generation",
        "standard",
    ),
    (
        "qwen3_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qk_norm": True,
        },
        "static-cache",
        "standard",
    ),
    (
        "phi3",
        {"partial_rotary_factor": 0.5},
        "text-generation",
        "standard",
    ),
    (
        "phi3",
        {"partial_rotary_factor": 0.5},
        "static-cache",
        "standard",
    ),
    (
        "gemma2",
        {
            "_config_cls": "Gemma2Config",
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
        },
        "text-generation",
        "standard",
    ),
    (
        "falcon",
        {"attn_qkv_bias": True},
        "text-generation",
        "standard",
    ),
    (
        "gpt2",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        "text-generation",
        "standard",
    ),
    (
        "bert",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        "feature-extraction",
        "standard",
    ),
    (
        "t5",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        "seq2seq",
        "standard",
    ),
    ("qwen3_5_vl", {}, "hybrid-qwen-vl", "qwen3_5_vl"),
    ("whisper", {}, "speech-to-text", "whisper"),
    ("mamba", {}, "ssm-text-generation", "mamba"),
]


def _display_key(model_type: str, task_name: str) -> str:
    """Build a unique display key for a model entry.

    Returns just ``model_type`` for the default ``text-generation`` task
    (backward compatible), or ``model_type (task_name)`` otherwise.
    """
    if task_name == "text-generation":
        return model_type
    return f"{model_type} ({task_name})"


# ------------------------------------------------------------------
# Detect affected model types from git diff
# ------------------------------------------------------------------


def _resolve_sha(ref: str) -> str:
    """Return the short SHA for *ref*, or *ref* itself if resolution fails.

    Falls back to the raw string on shallow clones, typos, or other git errors
    so the rest of the script can continue rather than crashing.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", ref],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ref


def _changed_files(base_ref: str, head_ref: str) -> list[str]:
    """Return file paths changed between *base_ref* and *head_ref*."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, head_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _affected_model_types(changed: list[str]) -> set[str]:
    """Determine which model types to diff.

    Heuristics:
    - ``components/`` or ``_configs.py`` changed → all models
    - ``models/<name>.py`` changed → that model type
    - ``tasks/`` or ``_exporter.py`` changed → all models
    """
    all_types = {m[0] for m in _DIFF_MODELS}
    affected: set[str] = set()

    for path in changed:
        # Normalise to forward slashes
        p = path.replace("\\", "/")
        if not p.startswith("src/mobius/"):
            continue

        rel = p[len("src/mobius/") :]

        if rel.startswith("components/"):
            return all_types
        if rel.startswith("tasks/"):
            return all_types
        if rel in ("_configs.py", "_exporter.py"):
            return all_types

        if rel.startswith("models/"):
            # e.g. "models/llama.py" → "llama"
            name = rel[len("models/") :]
            if name.endswith(".py"):
                name = name[: -len(".py")]
                # Match by prefix — e.g. "gemma" matches "gemma2"
                for mt in all_types:
                    if mt.startswith(name) or name.startswith(mt):
                        affected.add(mt)

    return affected


# ------------------------------------------------------------------
# Build canonical JSON for a single model at a given ref
# ------------------------------------------------------------------

_BUILDER_SCRIPT = textwrap.dedent("""\
    import json, sys
    sys.path.insert(0, "src")
    sys.path.insert(0, "tests")

    from _test_configs import (
        TINY_HIDDEN, TINY_INTERMEDIATE, TINY_HEADS, TINY_KV_HEADS,
        TINY_HEAD_DIM, TINY_LAYERS, TINY_VOCAB,
    )
    from mobius._graph_diff import canonicalize_graph

    entry = json.loads(sys.argv[1])
    model_type = entry["model_type"]
    overrides = entry["overrides"]
    task_name = entry["task_name"]
    build_kind = entry["build_kind"]

    def _base_config(config_cls=None, **kw):
        from mobius._configs import ArchitectureConfig
        if config_cls is None:
            config_cls = kw.pop("_config_cls", ArchitectureConfig)
        else:
            kw.pop("_config_cls", None)
        defaults = dict(
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            num_hidden_layers=TINY_LAYERS,
            vocab_size=TINY_VOCAB,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
        )
        defaults.update(kw)
        return config_cls(**defaults)

    def _build_standard():
        from mobius._registry import registry
        from mobius.tasks import CausalLMTask, get_task
        ov = dict(overrides)
        cls_name = ov.pop("_config_cls", None)
        if cls_name:
            import mobius._configs as _cfgs
            config = _base_config(getattr(_cfgs, cls_name), **ov)
        else:
            config = _base_config(**ov)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        if task_name == "static-cache":
            task = CausalLMTask(static_cache=True)
        else:
            task = get_task(task_name)
        return task.build(module, config)

    def _build_whisper():
        from mobius._configs import WhisperConfig
        from mobius._builder import build_from_module
        from mobius.models.whisper import (
            WhisperForConditionalGeneration,
        )
        from mobius.tasks import SpeechToTextTask
        config = WhisperConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="gelu", pad_token_id=0,
            tie_word_embeddings=True,
            attn_qkv_bias=True, attn_o_bias=True,
            encoder_layers=TINY_LAYERS,
            encoder_attention_heads=TINY_HEADS,
            encoder_ffn_dim=TINY_INTERMEDIATE,
            num_mel_bins=16, max_source_positions=100,
            max_target_positions=50, scale_embedding=True,
        )
        module = WhisperForConditionalGeneration(config)
        task = SpeechToTextTask()
        return build_from_module(module, config, task=task)

    def _build_mamba():
        from mobius._configs import MambaConfig
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask
        config = MambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,
            num_hidden_layers=TINY_LAYERS,
            state_size=8, conv_kernel=4, expand=2,
            time_step_rank=4, layer_norm_epsilon=1e-5,
            use_conv_bias=True, tie_word_embeddings=True,
        )
        module = MambaCausalLMModel(config)
        task = SSMCausalLMTask()
        return build_from_module(module, config, task=task)

    def _build_qwen3_5_vl():
        from mobius._configs import ArchitectureConfig, VisionConfig
        from mobius._registry import registry
        from mobius.tasks import get_task
        config = ArchitectureConfig(
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            num_hidden_layers=TINY_LAYERS,
            vocab_size=TINY_VOCAB,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
            attn_qk_norm=True,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention"],
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=248056,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_5_vl")
        module = model_cls(config)
        task = get_task("hybrid-qwen-vl")
        return task.build(module, config)

    if build_kind == "whisper":
        pkg = _build_whisper()
    elif build_kind == "mamba":
        pkg = _build_mamba()
    elif build_kind == "qwen3_5_vl":
        pkg = _build_qwen3_5_vl()
    else:
        pkg = _build_standard()

    result = {}
    for name, model in pkg.items():
        result[name] = canonicalize_graph(model.graph)
    print(json.dumps(result))
""")


def _build_at_ref(
    ref: str,
    model_type: str,
    overrides: dict,
    task_name: str,
    build_kind: str,
) -> dict[str, dict] | None:
    """Build model at *ref* and return canonical JSON per sub-model.

    Returns ``None`` if the build fails (e.g. the model didn't exist at
    that ref).
    """
    entry_json = json.dumps(
        {
            "model_type": model_type,
            "overrides": overrides,
            "task_name": task_name,
            "build_kind": build_kind,
        }
    )

    # Write builder script to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_BUILDER_SCRIPT)
        script_path = f.name

    try:
        # Run in a subprocess with the requested ref checked out via
        # git worktree so we don't disturb the working tree.
        with tempfile.TemporaryDirectory() as worktree:
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--detach",
                    worktree,
                    ref,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            try:
                # Copy the _graph_diff module into the worktree so
                # the builder script can import it even when the file
                # doesn't yet exist at the base ref.
                dest = Path(worktree) / "src" / "mobius" / "_graph_diff.py"
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(
                    _PROJECT_ROOT / "src" / "mobius" / "_graph_diff.py",
                    dest,
                )

                try:
                    result = subprocess.run(
                        [sys.executable, script_path, entry_json],
                        capture_output=True,
                        text=True,
                        cwd=worktree,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    print(
                        f"  ⚠ Build timed out for {model_type}@{ref}",
                        file=sys.stderr,
                    )
                    return None
                if result.returncode != 0:
                    print(
                        f"  ⚠ Build failed for {model_type}@{ref}:\n{result.stderr[:500]}",
                        file=sys.stderr,
                    )
                    return None
                return json.loads(result.stdout)
            finally:
                subprocess.run(
                    [
                        "git",
                        "worktree",
                        "remove",
                        "--force",
                        worktree,
                    ],
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "worktree", "prune"],
                    capture_output=True,
                )
    finally:
        os.unlink(script_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Architecture diff for ONNX GenAI models")
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Git ref for the base (e.g. HEAD~1 or origin/main)",
    )
    parser.add_argument(
        "--head-ref",
        required=True,
        help="Git ref for the head (e.g. HEAD)",
    )
    parser.add_argument(
        "--output",
        default="arch_diff.md",
        help="Output Markdown file (default: arch_diff.md)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Diff all models regardless of changed files",
    )
    args = parser.parse_args()

    from mobius._graph_diff import (
        diff_graphs,
        render_markdown,
    )

    # Resolve refs to short SHAs for display in the markdown output
    base_sha = _resolve_sha(args.base_ref)
    head_sha = _resolve_sha(args.head_ref)

    # 1. Detect affected models
    if args.all:
        affected = {m[0] for m in _DIFF_MODELS}
    else:
        changed = _changed_files(args.base_ref, args.head_ref)
        affected = _affected_model_types(changed)
        if not affected:
            print("No model source files changed — nothing to diff.")
            # Write a minimal report
            md = render_markdown(
                {}, base_ref=base_sha, head_ref=head_sha, repo_url=_GITHUB_REPO_URL
            )
            Path(args.output).write_text(md, encoding="utf-8")
            return

    print(f"Affected models: {sorted(affected)}")

    # 2. Build + diff each affected model
    all_diffs: dict[str, dict[str, list]] = {}
    failed_head_builds: list[str] = []

    for model_type, overrides, task_name, build_kind in _DIFF_MODELS:
        if model_type not in affected:
            continue

        display = _display_key(model_type, task_name)
        print(f"Building {display} at {args.base_ref} …")
        base_pkg = _build_at_ref(
            args.base_ref,
            model_type,
            overrides,
            task_name,
            build_kind,
        )
        print(f"Building {display} at {args.head_ref} …")
        head_pkg = _build_at_ref(
            args.head_ref,
            model_type,
            overrides,
            task_name,
            build_kind,
        )

        if head_pkg is None:
            # Head-ref (current code) build failure is always an error.
            failed_head_builds.append(display)
            continue

        if base_pkg is None:
            # Base-ref failure is expected for newly added models.
            # Skip the diff but don't treat it as an error.
            continue

        sub_models: dict[str, dict] = {}
        all_sub_names = sorted(set(base_pkg.keys()) | set(head_pkg.keys()))
        for sub_name in all_sub_names:
            base_canon = base_pkg.get(
                sub_name,
                _empty_canonical(),
            )
            head_canon = head_pkg.get(
                sub_name,
                _empty_canonical(),
            )
            changes = diff_graphs(base_canon, head_canon)
            sub_models[sub_name] = {
                "changes": changes,
                "_base_ops": base_canon.get("op_sequence", []),
                "_head_ops": head_canon.get("op_sequence", []),
                "_base_node_count": len(base_canon.get("nodes", [])),
                "_head_node_count": len(head_canon.get("nodes", [])),
            }
        all_diffs[display] = sub_models

    # 3. Render and write
    md = render_markdown(
        all_diffs, base_ref=base_sha, head_ref=head_sha, repo_url=_GITHUB_REPO_URL
    )
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")

    # 4. Fail if any head-ref builds failed
    if failed_head_builds:
        print(
            f"\n✗ {len(failed_head_builds)} model(s) failed to build at {args.head_ref}:",
            file=sys.stderr,
        )
        for name in failed_head_builds:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)


def _empty_canonical() -> dict:
    """Return an empty canonical form for a missing sub-model."""
    return {
        "interface": {"inputs": [], "outputs": []},
        "initializers": [],
        "nodes": [],
        "op_sequence": [],
    }


if __name__ == "__main__":
    main()
