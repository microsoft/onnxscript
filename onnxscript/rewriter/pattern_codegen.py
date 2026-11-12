# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Generate rewrite-rule source code from a subgraph of an existing model.

Writing the ``pattern`` of a rewrite rule by hand means transcribing a piece of a model
into ``op.Xxx(...)`` calls. This tool does the transcription:

1. The user names the nodes of interest (``ir.Node.name``).
2. The named nodes are turned into an :class:`ir.GraphView <onnx_ir.GraphView>`. Values
   produced outside the selection become pattern inputs. Values that escape the selection
   (used by other nodes or graph outputs) become pattern outputs.
3. Source code for a :class:`~onnxscript.rewriter.pattern.RewriteRuleClassBase` subclass is
   generated with libcst. The ``rewrite`` method is a stub for the user to fill in.

Command line usage::

    python -m onnxscript.rewriter.pattern_codegen model.onnx --nodes Add_1 Relu_2

Python usage::

    source = pattern_codegen.generate_rewrite_rule(model, ["Add_1", "Relu_2"])

The generated pattern is intentionally specific (it keeps op domains, scalar and list
attributes, and the output count of multi-output nodes). Generalize it by hand, for example
by replacing an attribute value with a pattern variable.

libcst is required for code generation (``pip install libcst``).
"""

from __future__ import annotations

__all__ = [
    "generate_pattern_source",
    "generate_rewrite_rule",
    "select_subgraph",
]

import argparse
import difflib
import json
import keyword
import math
import re
import sys
from collections.abc import Collection, Sequence
from typing import Any

import onnx_ir as ir

_RESERVED_NAMES = frozenset({"self", "op", "pattern", "ir"})
_DEFAULT_DOMAINS = frozenset({"", "ai.onnx"})
_MAX_CLASS_NAME_NODES = 4


def _import_libcst():
    try:
        import libcst
    except ImportError as e:
        raise ImportError(
            "Generating rewrite rules requires libcst. Install it with 'pip install libcst'."
        ) from e
    return libcst


# ---------------------------------------------------------------------------
# Step 1 and 2: select nodes by name and build a graph view.
# ---------------------------------------------------------------------------


def select_subgraph(graph: ir.Graph | ir.Model, node_names: Collection[str]) -> ir.GraphView:
    """Create a graph view over the nodes called ``node_names``.

    The view's inputs are the values consumed by the selected nodes but produced elsewhere
    (graph inputs, initializers and outputs of unselected nodes), in order of first use.
    The view's outputs are the selected nodes' values that are used by an unselected node
    or are graph outputs. A value that nothing consumes is also an output, which keeps a
    selection that ends in a dangling node well defined.

    Args:
        graph: The graph (or the main graph of a model) that contains the nodes.
        node_names: Names of the nodes to select. Names must be unique in the graph.

    Returns:
        A :class:`ir.GraphView <onnx_ir.GraphView>` of the selected nodes, in graph order.

    Raises:
        ValueError: If ``node_names`` is empty, a name is missing or ambiguous, or the
            selection is not convex. A selection is not convex when a path leaves it and
            comes back, because replacing it would create a cycle.
    """
    if isinstance(graph, ir.Model):
        graph = graph.graph
    wanted = list(dict.fromkeys(node_names))
    if not wanted:
        raise ValueError("At least one node name is required.")

    by_name: dict[str, list[ir.Node]] = {}
    for node in graph:
        if node.name:
            by_name.setdefault(node.name, []).append(node)

    missing = [name for name in wanted if name not in by_name]
    if missing:
        hints = []
        for name in missing:
            close = difflib.get_close_matches(name, by_name, n=3)
            hints.append(f"{name!r}" + (f" (did you mean {close}?)" if close else ""))
        raise ValueError(f"Node names not found in the graph: {', '.join(hints)}")
    ambiguous = [name for name in wanted if len(by_name[name]) > 1]
    if ambiguous:
        raise ValueError(f"Node names are not unique in the graph: {ambiguous}")

    selected_names = set(wanted)
    selected = [node for node in graph if node.name in selected_names]
    selected_set = set(selected)
    _check_convex(selected, selected_set)

    inputs: list[ir.Value] = []
    seen_inputs: set[ir.Value] = set()
    for node in selected:
        for value in node.inputs:
            if value is None or value in seen_inputs:
                continue
            producer = value.producer()
            if producer is None or producer not in selected_set:
                seen_inputs.add(value)
                inputs.append(value)

    outputs: list[ir.Value] = []
    for node in selected:
        for value in node.outputs:
            uses = value.uses()
            escapes = value.is_graph_output() or any(
                use.node not in selected_set for use in uses
            )
            if escapes or not uses:
                outputs.append(value)

    return ir.GraphView(inputs, outputs, nodes=selected, name="pattern")


def _check_convex(selected: Sequence[ir.Node], selected_set: set[ir.Node]) -> None:
    """Raise if a path from a selected node leaves the selection and re-enters it."""
    visited: set[ir.Node] = set()
    stack = [
        use.node
        for node in selected
        for value in node.outputs
        for use in value.uses()
        if use.node not in selected_set
    ]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for value in node.outputs:
            for use in value.uses():
                if use.node in selected_set:
                    raise ValueError(
                        f"The selected nodes are not convex: node {use.node.name!r} depends "
                        f"on node {node.name!r}, which is itself downstream of the "
                        "selection. Add the nodes in between to the selection."
                    )
                stack.append(use.node)


# ---------------------------------------------------------------------------
# Step 3: code generation with libcst.
# ---------------------------------------------------------------------------


class _Namer:
    """Assigns a unique, valid Python identifier to each ir.Value."""

    def __init__(self) -> None:
        self._taken: set[str] = set(_RESERVED_NAMES)
        self._names: dict[ir.Value, str] = {}

    def name_of(self, value: ir.Value, fallback: str) -> str:
        if value in self._names:
            return self._names[value]
        base = re.sub(r"\W+", "_", value.name or "").strip("_")
        if not base:
            base = fallback
        if base[0].isdigit():
            base = f"v_{base}"
        if keyword.iskeyword(base):
            base = f"{base}_"
        name, i = base, 1
        while name in self._taken:
            name = f"{base}_{i}"
            i += 1
        self._taken.add(name)
        self._names[value] = name
        return name


def _literal_source(value: object) -> str | None:
    """Python source for an attribute value, or None if the pattern language cannot express it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'float("inf")' if value > 0 else 'float("-inf")'
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Sequence):
        items = [_literal_source(item) for item in value]
        if any(item is None for item in items):
            return None
        return f"[{', '.join(items)}]"  # type: ignore[arg-type]
    return None


_SUPPORTED_ATTR_TYPES = frozenset(
    {
        ir.AttributeType.INT,
        ir.AttributeType.FLOAT,
        ir.AttributeType.STRING,
        ir.AttributeType.INTS,
        ir.AttributeType.FLOATS,
        ir.AttributeType.STRINGS,
    }
)


def _kwarg(cst, name: str, value):
    """A keyword argument rendered as ``name=value``, without spaces around ``=``."""
    return cst.Arg(
        value=value,
        keyword=cst.Name(name),
        equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
        ),
    )


def _node_call(cst, node: ir.Node, arg_names: Sequence[str | None], notes: list[str]):
    """Build the ``op.Type(...)`` call expression for ``node``."""
    if not node.op_type.isidentifier():
        raise ValueError(
            f"Cannot generate code for op type {node.op_type!r} of {node.name!r}."
        )

    # Trailing absent optional inputs are implied, so they are not spelled out.
    names = list(arg_names)
    while names and names[-1] is None:
        names.pop()
    args = [
        cst.Arg(value=cst.Name("None") if name is None else cst.Name(name)) for name in names
    ]

    for attr_name, attr in node.attributes.items():
        source = None
        if not attr.is_ref() and attr.type in _SUPPORTED_ATTR_TYPES:
            source = _literal_source(attr.value)
        if source is None:
            notes.append(
                f"Attribute {attr_name!r} of {node.op_type} ({node.name}) is not matched "
                "by this pattern."
            )
            continue
        args.append(_kwarg(cst, attr_name, cst.parse_expression(source)))

    if node.domain not in _DEFAULT_DOMAINS:
        args.append(_kwarg(cst, "_domain", cst.SimpleString(json.dumps(node.domain))))
    if len(node.outputs) != 1:
        args.append(_kwarg(cst, "_outputs", cst.Integer(str(len(node.outputs)))))
    return cst.Call(func=cst.Attribute(cst.Name("op"), cst.Name(node.op_type)), args=args)


def _name_or_tuple(cst, names: Sequence[str]):
    if len(names) == 1:
        return cst.Name(names[0])
    return cst.Tuple(elements=[cst.Element(cst.Name(n)) for n in names], lpar=[], rpar=[])


def _comment_lines(cst, notes: Sequence[str]):
    return [cst.EmptyLine(comment=cst.Comment(f"# {note}")) for note in notes]


def _docstring_text(node_names: str) -> str:
    """Text safe to place inside a one-line triple-quoted docstring."""
    text = " ".join(f"Rewrite rule generated from nodes: {node_names}".split())
    return text.replace("\\", "/").replace('"', "'")


def _default_class_name(nodes: Sequence[ir.Node]) -> str:
    op_types = [node.op_type for node in nodes]
    if len(op_types) > _MAX_CLASS_NAME_NODES:
        return "GeneratedRule"
    return "".join(op_types) + "Rule"


def generate_pattern_source(subgraph: ir.GraphView, *, class_name: str | None = None) -> str:
    """Generate the source code of a rewrite rule class for ``subgraph``.

    Args:
        subgraph: The view to transcribe, usually from :func:`select_subgraph`.
        class_name: Name of the generated class. Defaults to the concatenated op types
            of the nodes followed by ``Rule``.

    Returns:
        Python source for a module defining a ``pattern.RewriteRuleClassBase`` subclass.
        Its ``pattern`` method reproduces the subgraph and its ``rewrite`` method raises
        ``NotImplementedError``.

    Raises:
        ImportError: If libcst is not installed.
        ValueError: If ``class_name`` is not a valid identifier.
    """
    cst = _import_libcst()

    nodes = list(subgraph)
    if class_name is None:
        class_name = _default_class_name(nodes)
    if not class_name.isidentifier() or keyword.iskeyword(class_name):
        raise ValueError(f"Invalid class name: {class_name!r}")

    namer = _Namer()
    input_names = [
        namer.name_of(value, f"input_{i}") for i, value in enumerate(subgraph.inputs)
    ]

    notes: list[str] = []
    statements: list[Any] = []
    produced = 0
    for node in nodes:
        arg_names = [
            None if value is None else namer.name_of(value, "")  # inputs are already named
            for value in node.inputs
        ]
        call = _node_call(cst, node, arg_names, notes)
        target_names = []
        for value in node.outputs:
            target_names.append(namer.name_of(value, f"v{produced}"))
            produced += 1
        statements.append(
            cst.SimpleStatementLine(
                [
                    cst.Assign(
                        targets=[cst.AssignTarget(_name_or_tuple(cst, target_names))],
                        value=call,
                    )
                ],
                leading_lines=_comment_lines(cst, notes),
            )
        )
        notes.clear()

    output_names = [namer.name_of(value, "") for value in subgraph.outputs]
    statements.append(
        cst.SimpleStatementLine(
            [
                cst.Return(
                    _name_or_tuple(cst, output_names),
                    whitespace_after_return=cst.SimpleWhitespace(" "),
                )
            ]
        )
    )

    def params(annotation: bool) -> Any:
        def make(name: str) -> Any:
            return cst.Param(
                cst.Name(name),
                annotation=cst.Annotation(cst.parse_expression("ir.Value"))
                if annotation
                else None,
            )

        return cst.Parameters(
            params=[cst.Param(cst.Name("self")), cst.Param(cst.Name("op"))]
            + [make(name) for name in input_names]
        )

    node_names = ", ".join(node.name or node.op_type for node in nodes)
    docstring = cst.SimpleStatementLine(
        [cst.Expr(cst.SimpleString('"""' + _docstring_text(node_names) + '"""'))]
    )
    pattern_fn = cst.FunctionDef(
        name=cst.Name("pattern"),
        params=params(annotation=False),
        body=cst.IndentedBlock(statements),
        leading_lines=[cst.EmptyLine(indent=False)],
    )
    rewrite_fn = cst.FunctionDef(
        name=cst.Name("rewrite"),
        params=params(annotation=True),
        body=cst.IndentedBlock(
            [
                cst.SimpleStatementLine(
                    [
                        cst.Raise(
                            cst.Call(
                                cst.Name("NotImplementedError"),
                                [cst.Arg(cst.SimpleString('"Write the replacement here."'))],
                            )
                        )
                    ]
                )
            ]
        ),
        leading_lines=[cst.EmptyLine(indent=False)],
    )
    class_def = cst.ClassDef(
        name=cst.Name(class_name),
        bases=[cst.Arg(cst.Attribute(cst.Name("pattern"), cst.Name("RewriteRuleClassBase")))],
        body=cst.IndentedBlock([docstring, pattern_fn, rewrite_fn]),
        leading_lines=[cst.EmptyLine(indent=False), cst.EmptyLine(indent=False)],
    )
    module = cst.Module(
        body=[
            cst.parse_statement("from __future__ import annotations"),
            cst.parse_statement("import onnx_ir as ir").with_changes(
                leading_lines=[cst.EmptyLine(indent=False)]
            ),
            cst.parse_statement("from onnxscript.rewriter import pattern").with_changes(
                leading_lines=[cst.EmptyLine(indent=False)]
            ),
            class_def,
        ]
    )
    return module.code


def generate_rewrite_rule(
    model: ir.Model | ir.Graph,
    node_names: Collection[str],
    *,
    class_name: str | None = None,
) -> str:
    """Generate rewrite-rule source code for the nodes called ``node_names``.

    Equivalent to ``generate_pattern_source(select_subgraph(model, node_names))``.
    """
    return generate_pattern_source(select_subgraph(model, node_names), class_name=class_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a rewrite rule skeleton from nodes of an ONNX model."
    )
    parser.add_argument("model", help="Path to the ONNX model.")
    parser.add_argument("--nodes", nargs="+", required=True, help="Names of the nodes.")
    parser.add_argument("--class-name", default=None, help="Name of the generated class.")
    parser.add_argument("-o", "--output", default=None, help="Write to this file, not stdout.")
    args = parser.parse_args(argv)

    model = ir.load(args.model)
    try:
        source = generate_rewrite_rule(model, args.nodes, class_name=args.class_name)
    except (ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(source)
    else:
        print(source, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
