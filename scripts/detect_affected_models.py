#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Detect which model_types are affected by a set of changed files.

Uses AST-based static import analysis (no actual imports) to determine
which model types need testing when source files change. Designed for
CI diff-based scoping — outputs JSON for downstream workflow consumption.

Usage::

    # From git diff
    git diff --name-only origin/main...HEAD | \\
        python scripts/detect_affected_models.py --stdin

    # Explicit file list
    python scripts/detect_affected_models.py \\
        --changed-files "src/mobius/models/qwen2.py
    src/mobius/models/llama.py"

    # Output: {"affected": ["llama", "qwen2", ...], "run_all": false}
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src" / "mobius"

# ----------------------------------------------------------------
# Shared infrastructure paths — any change triggers run_all
# ----------------------------------------------------------------
_SHARED_INFRA_PATTERNS = (
    "src/mobius/_configs.py",
    "src/mobius/_registry.py",
    "src/mobius/_builder.py",
    "src/mobius/_weight_loading.py",
    "src/mobius/_model_package.py",
    "src/mobius/_exporter.py",
    # Re-export hub for all model classes — any change here affects all models
    "src/mobius/models/__init__.py",
)

_SHARED_INFRA_PREFIXES = (
    "src/mobius/components/",
    "src/mobius/tasks/",
)


def classify_file(path: str) -> str:
    """Classify a changed file path.

    Returns one of: 'model', 'component', 'task', 'shared_infra',
    'test', 'other'.
    """
    normalized = path.replace("\\", "/")

    if not normalized.startswith("src/mobius/"):
        # Test infrastructure files that affect all models
        if normalized in (
            "tests/conftest.py",
            "tests/_test_configs.py",
        ):
            return "shared_infra"
        if normalized.endswith("_test.py") or normalized.startswith("tests/"):
            return "test"
        return "other"

    rel = normalized[len("src/mobius/") :]

    # Shared infrastructure patterns
    if normalized in _SHARED_INFRA_PATTERNS:
        return "shared_infra"
    for prefix in _SHARED_INFRA_PREFIXES:
        if normalized.startswith(prefix):
            return "shared_infra"

    # Model files
    if rel.startswith("models/") and not rel.endswith("_test.py"):
        return "model"

    # Test files within the source tree
    if rel.endswith("_test.py"):
        return "test"

    return "other"


# ----------------------------------------------------------------
# AST-based import analysis
# ----------------------------------------------------------------


def _parse_imports(filepath: Path) -> set[str]:
    """Extract imported module names from a Python file using AST.

    Returns a set of dotted module names that appear in import
    statements. Only collects imports from within the
    mobius package.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mobius"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("mobius"):
                imports.add(node.module)
    return imports


def _module_name_from_path(filepath: Path) -> str | None:
    """Convert a file path to a dotted module name.

    Returns None if the file is not under the src/ tree.
    """
    try:
        rel = filepath.resolve().relative_to(_PROJECT_ROOT / "src")
    except ValueError:
        return None

    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def _build_import_graph(
    search_dir: Path,
) -> dict[str, set[str]]:
    """Build a module → set[imported_modules] graph for all .py files.

    The graph maps each module name to the set of mobius
    modules it directly imports.
    """
    graph: dict[str, set[str]] = {}
    for pyfile in search_dir.rglob("*.py"):
        if pyfile.name.startswith("__"):
            continue
        if pyfile.name.endswith("_test.py"):
            continue
        mod_name = _module_name_from_path(pyfile)
        if mod_name:
            graph[mod_name] = _parse_imports(pyfile)
    return graph


def _find_reverse_dependents(
    target_module: str,
    import_graph: dict[str, set[str]],
) -> set[str]:
    """Find all modules that transitively depend on target_module.

    Uses BFS on the reverse dependency graph.
    """
    # Build reverse graph: module → set of modules that import it
    reverse: dict[str, set[str]] = {}
    for mod, deps in import_graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(mod)

    visited: set[str] = set()
    queue: collections.deque[str] = collections.deque([target_module])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for dependent in reverse.get(current, set()):
            if dependent not in visited:
                queue.append(dependent)

    # Don't include the target itself
    visited.discard(target_module)
    return visited


def _model_file_to_module(rel_path: str) -> str | None:
    """Convert a relative model file path to a module name.

    Example: 'models/qwen.py' → 'mobius.models.qwen'
    """
    if not rel_path.endswith(".py"):
        return None
    module_path = rel_path[:-3].replace("/", ".")
    return f"mobius.{module_path}"


# ----------------------------------------------------------------
# Registry mapping: source module → model_types
#
# The registry imports classes from mobius.models (the
# package __init__), but the actual definitions live in submodules
# like models.base, models.falcon, etc. We parse models/__init__.py
# to resolve class → source submodule, then parse _registry.py to
# map class → model_types. Combined: submodule → model_types.
# ----------------------------------------------------------------


def _build_class_to_source_module() -> dict[str, str]:
    """Parse models/__init__.py to map class names to source submodules.

    E.g. CausalLMModel → mobius.models.base
         FalconCausalLMModel → mobius.models.falcon
    """
    init_file = _SRC_ROOT / "models" / "__init__.py"
    class_to_source: dict[str, str] = {}

    try:
        source = init_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(init_file))
    except (SyntaxError, UnicodeDecodeError):
        return class_to_source

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("mobius.models."):
                for alias in node.names:
                    class_to_source[alias.name] = node.module

    return class_to_source


def _build_registry_class_to_types() -> dict[str, list[str]]:
    """Parse _registry.py to map class names to registered model_types.

    Handles three patterns:
    1. Direct: reg.register("name", ClassName)
    2. For-loop: for name in (...): reg.register(name, ClassName)
    3. Dict-loop: for name, cls in {...}.items(): reg.register(name, cls)
    """
    registry_file = _SRC_ROOT / "_registry.py"
    class_to_types: dict[str, list[str]] = {}

    try:
        source = registry_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(registry_file))
    except (SyntaxError, UnicodeDecodeError):
        return class_to_types

    for node in ast.walk(tree):
        # Pattern 1: Direct reg.register("name", ClassName)
        if isinstance(node, ast.Call):
            cls_name, arch_name = _match_register_call(node)
            if cls_name and arch_name:
                class_to_types.setdefault(cls_name, []).append(arch_name)

        # Pattern 2 & 3: For-loop with reg.register in body
        if isinstance(node, ast.For):
            _process_for_loop(node, class_to_types)

    return {c: sorted(set(t)) for c, t in class_to_types.items()}


def _match_register_call(
    node: ast.Call,
) -> tuple[str | None, str | None]:
    """Match a reg.register("name", ClassName) call.

    Returns (class_name, arch_name) or (None, None).
    """
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "reg"
    ):
        return None, None
    if len(node.args) < 2:
        return None, None

    name_node = node.args[0]
    cls_node = node.args[1]

    if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
        return None, None
    if not isinstance(cls_node, ast.Name):
        return None, None

    return cls_node.id, name_node.value


def _process_for_loop(
    node: ast.For,
    class_to_types: dict[str, list[str]],
) -> None:
    """Extract model_type → class mappings from for-loop patterns."""
    # Pattern 2: for name in ("llama", "qwen2", ...): reg.register(name, Cls)
    string_names = _extract_string_constants(node.iter)
    if string_names:
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr):
                continue
            call = stmt.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register"
                and isinstance(func.value, ast.Name)
                and func.value.id == "reg"
            ):
                continue
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Name):
                cls_name = call.args[1].id
                class_to_types.setdefault(cls_name, []).extend(string_names)
        return

    # Pattern 3: for name, cls in {...}.items(): reg.register(name, cls)
    iter_node = node.iter
    if (
        isinstance(iter_node, ast.Call)
        and isinstance(iter_node.func, ast.Attribute)
        and iter_node.func.attr == "items"
        and isinstance(iter_node.func.value, ast.Dict)
    ):
        dict_node = iter_node.func.value
        for key, value in zip(dict_node.keys, dict_node.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Name)
            ):
                class_to_types.setdefault(value.id, []).append(key.value)


def _extract_string_constants(node: ast.expr) -> list[str]:
    """Extract string constants from a Tuple or List AST node."""
    if isinstance(node, (ast.Tuple, ast.List)):
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                result.append(elt.value)
        return result
    return []


def _build_source_module_to_types() -> dict[str, list[str]]:
    """Build the final source_module → [model_types] mapping.

    Combines __init__.py class→source resolution with _registry.py
    class→model_types mapping to produce submodule→model_types.
    """
    class_to_source = _build_class_to_source_module()
    class_to_types = _build_registry_class_to_types()

    # Also collect direct imports from _registry.py itself
    # (for classes imported from submodules directly, not via __init__)
    registry_file = _SRC_ROOT / "_registry.py"
    try:
        source = registry_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(registry_file))
    except (SyntaxError, UnicodeDecodeError):
        tree = None

    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("mobius.models."):
                    for alias in node.names:
                        # Only add if not already resolved
                        if alias.name not in class_to_source:
                            class_to_source[alias.name] = node.module

    # Now combine: source_module → model_types
    module_to_types: dict[str, list[str]] = {}
    for cls_name, types in class_to_types.items():
        source_mod = class_to_source.get(cls_name)
        if source_mod:
            module_to_types.setdefault(source_mod, []).extend(types)

    return {m: sorted(set(t)) for m, t in module_to_types.items()}


# ----------------------------------------------------------------
# Main detection logic
# ----------------------------------------------------------------


def detect_affected_models(
    changed_files: list[str],
) -> dict[str, list[str] | bool]:
    """Determine which model_types are affected by file changes.

    Args:
        changed_files: List of file paths relative to repository root.

    Returns:
        Dict with keys:
            - "affected": sorted list of affected model_type strings
            - "run_all": True if all models should be tested
    """
    affected: set[str] = set()
    run_all = False

    # Classify files
    model_files: list[str] = []
    for path in changed_files:
        category = classify_file(path)
        if category == "shared_infra":
            run_all = True
            break
        elif category == "model":
            # Deleted model files could break dependents — run all
            full_path = _PROJECT_ROOT / path
            if not full_path.exists():
                run_all = True
                break
            model_files.append(path)

    if run_all:
        return {"affected": [], "run_all": True}

    if not model_files:
        return {"affected": [], "run_all": False}

    # Build the registry map: source_module → [model_types]
    registry_map = _build_source_module_to_types()

    # Build import graph for transitive analysis
    import_graph = _build_import_graph(_SRC_ROOT)

    for path in model_files:
        normalized = path.replace("\\", "/")
        rel = normalized[len("src/mobius/") :]
        module_name = _model_file_to_module(rel)
        if not module_name:
            continue

        # Direct: this module's own registered types
        if module_name in registry_map:
            affected.update(registry_map[module_name])

        # Transitive: find modules that import from this model file
        dependents = _find_reverse_dependents(module_name, import_graph)
        for dep_module in dependents:
            if dep_module in registry_map:
                affected.update(registry_map[dep_module])

    return {"affected": sorted(affected), "run_all": False}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect model_types affected by changed files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--changed-files",
        help="Newline-separated list of changed file paths",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read changed file paths from stdin (one per line)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="json",
        help=(
            "Output format. 'json' prints the full result dict. "
            "'github' sets GitHub Actions output variables."
        ),
    )
    args = parser.parse_args()

    if args.stdin:
        changed = [line.strip() for line in sys.stdin if line.strip()]
    else:
        changed = [line.strip() for line in args.changed_files.split("\n") if line.strip()]

    result = detect_affected_models(changed)

    if args.output_format == "github":
        # Output for GitHub Actions
        affected_json = json.dumps(result["affected"])
        run_all = "true" if result["run_all"] else "false"
        has_affected = "true" if (result["run_all"] or result["affected"]) else "false"
        print(f"affected={affected_json}")
        print(f"run_all={run_all}")
        print(f"has_affected={has_affected}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
