# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Rewrite rules for ONNX models."""

from __future__ import annotations

import abc
import dataclasses
import itertools
from typing import (
    Callable,
    Sequence,
    TypeVar,
)

import onnxscript.optimizer
import onnxscript.rewriter._basics as _basics
import onnxscript.rewriter._matcher as _matcher
import onnxscript.rewriter._pattern_ir as _pattern_ir
from onnxscript import ir
from onnxscript.ir import _tape, convenience

T = TypeVar("T")

RewriterContext = _tape.Builder


@dataclasses.dataclass
class ReplacementSubgraph:
    """A subgraph that will replace the matched pattern."""

    match: _basics.MatchResult
    new_outputs: Sequence[ir.Value]
    new_nodes: Sequence[ir.Node]
    new_initializers: Sequence[ir.Value]
    used_opsets: _tape.UsedOpsets


def always_true(*args, **kwargs) -> bool:
    """A condition function that always returns True.

    This is used when no condition function is provided for a rewrite rule.
    """
    return True


class ReplacementPatternFunction:
    """The replacement pattern that will replace the targeted pattern.

    Attributes:
        function (Callable): The replacement function that will be used to replace the matched pattern.
    """

    def __init__(self, function) -> None:
        self._function = function

    def get_replacement(self, match: _basics.MatchResult) -> ReplacementSubgraph | None:
        context = RewriterContext()
        new_outputs = self._function(context, **match.bindings)
        if new_outputs is None:
            return None  # Failed to create replacement subgraph
        if not isinstance(new_outputs, Sequence):
            new_outputs = [new_outputs]
        return ReplacementSubgraph(
            match, new_outputs, context.nodes, context.initializers, context.used_opsets
        )


def _update_opset_imports(
    graph_or_function: ir.Graph | ir.Function, delta: ReplacementSubgraph
):
    imports = graph_or_function.opset_imports
    for domain, version in delta.used_opsets:
        if domain not in imports:
            # use 1 as default version if not explicitly specified
            imports[domain] = version if version is not None else 1
        elif version is not None and version != imports[domain]:
            raise ValueError(
                f"Multiple versions of opset {domain} used. "
                f"Expected version {imports[domain]}, but got {version}."
            )


class RewriteRule:
    def __init__(
        self,
        target_pattern: _pattern_ir.GraphPattern | Callable,
        replacement_pattern: ReplacementPatternFunction | Callable,
        condition_function: Callable | None = None,
        matcher: _matcher.PatternMatcher
        | Callable[[_pattern_ir.GraphPattern], _matcher.PatternMatcher]
        | None = None,
        verbose: int = 0,
        name: str | None = None,
        remove_nodes: bool = True,
        graph_pre_visitor: Callable[[], None] | None = None,
        graph_post_visitor: Callable[[], None] | None = None,
        as_function: bool = False,
    ) -> None:
        """Create a rewrite rule.

        Args:
            target_pattern: The _pattern_ir.GraphPattern that will be matched against the IR.
                If a callable is provided, it will be converted to a _pattern_ir.GraphPattern.
            replacement_pattern: The ReplacementPatternFunction that will be used to
                replace the matched pattern. If a callable is provided, it will be
                converted to a ReplacementPatternFunction.
            condition_function: The condition function that will be used to check if
                the pattern match found should be rewritten.
            matcher: The pattern matcher that will be used to match the pattern.
                If not provided, a default matcher will be used.
            verbose: The verbosity level of the rule.
            name: An optional name for the pattern that will show up in verbose logging.
            remove_nodes: If True, the matched nodes will be removed from the graph.
            graph_pre_visitor: A function that will be called before applying the
                rewriting to the top-level graph or a function.
            graph_post_visitor: A function that will be called after the rewriting
                is complete for a graph or function.
            as_function: If True, the matched nodes will be extracted into a model
                local function. This is only supported when remove_nodes=True and
                when the replacement subgraph has a single node, representing the
                function call.
        """
        if as_function and not remove_nodes:
            raise ValueError("as_function=True is only supported when remove_nodes=True.")
        if not isinstance(target_pattern, _pattern_ir.GraphPattern):
            target_pattern = _pattern_ir._to_graph_pattern(target_pattern)
        self._target_pattern = target_pattern

        if not isinstance(replacement_pattern, ReplacementPatternFunction):
            replacement_pattern = ReplacementPatternFunction(replacement_pattern)
        self._replacement_pattern = replacement_pattern
        self._condition_function = condition_function or always_true
        if isinstance(matcher, _matcher.PatternMatcher):
            self._matcher = matcher
        elif matcher is None:
            if target_pattern.has_single_output_node:
                self._matcher = _matcher.SimplePatternMatcher(self._target_pattern)
            else:
                import onnxscript.rewriter.generic_pattern as generic_pattern

                self._matcher = generic_pattern.GenericPatternMatcher(self._target_pattern)
        else:
            self._matcher = matcher(self._target_pattern)
        self._verbose = verbose
        self.name = name
        self.remove_nodes = remove_nodes
        self.graph_pre_visitor = graph_pre_visitor
        self.graph_post_visitor = graph_post_visitor
        self.as_function = as_function

    def __str__(self) -> str:
        return self.name if self.name else "Anonymous Rule"

    def try_rewrite(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        *,
        verbose: int | None = None,
        tracer: _basics.MatchingTracer | None = None,
    ) -> ReplacementSubgraph | None:
        """If the node matches the pattern, then replace the node with the replacement pattern."""
        if verbose and verbose > 2:
            print(f"[try_rewrite] {self}")
        verbose = verbose if verbose is not None else self._verbose
        match = self._matcher.match(
            model, graph_or_function, node, verbose=verbose, remove_nodes=self.remove_nodes
        )
        if match:
            context = None  # TODO(rama)
            for var in self._target_pattern.inputs:
                if var.name is not None:
                    if var.name not in match.bindings:
                        match.bind(var.name, None)
            check_match_result = self._condition_function(context, **match.bindings)
            if not check_match_result:
                # If check function was provided, but it failed, return the reason for failure to the tracer.
                if isinstance(check_match_result, _basics.MatchResult):
                    match.fail(
                        check_match_result.reason,
                        check_match_result.failure_nodes_and_values,
                    )
                if tracer:
                    tracer.log(
                        self,
                        graph_or_function,
                        node,
                        match,
                        _basics.MatchStatus.CONDITION_FAILED,
                    )
                return None
            replacement_subgraph = self._replacement_pattern.get_replacement(match)
            if replacement_subgraph is None:
                if tracer:
                    tracer.log(
                        self,
                        graph_or_function,
                        node,
                        match,
                        _basics.MatchStatus.REPLACEMENT_FAILED,
                    )
                return None
            if len(replacement_subgraph.new_outputs) != self._target_pattern.num_outputs:
                raise ValueError(
                    f"Number of outputs from replacement function does not match the number of outputs from the target pattern. "
                    f"Expected {self._target_pattern.num_outputs}, but got {len(replacement_subgraph.new_outputs)}."
                )
            # TODO(rama): Remove the opset imports from deleted nodes?
            _update_opset_imports(graph_or_function, replacement_subgraph)
            _update_opset_imports(model.graph, replacement_subgraph)
            if tracer:
                tracer.log(self, graph_or_function, node, match, _basics.MatchStatus.SUCCESS)
            return replacement_subgraph
        if tracer:
            tracer.log(self, graph_or_function, node, match, _basics.MatchStatus.NO_MATCH)
        return None

    def apply_to_model(
        self,
        model: ir.Model,
        *,
        commute: bool = False,
        verbose: int | None = None,
        tracer: _basics.MatchingTracer | None = None,
    ):
        # A convenience method to apply the rule to a model. We use a RewriteRuleSet to
        # handle commutative rules.
        return RewriteRuleSet([self], commute=commute).apply_to_model(
            model, verbose=verbose, tracer=tracer
        )

    def commute(self) -> Sequence[RewriteRule]:
        def replace_pattern(new_pattern):
            """Return a shallow copy of self with node_pattern replaced by new_pattern."""
            # TODO(rama): Maybe we should use a better alternative to construct new matcher.
            matcher_class = type(self._matcher)
            return RewriteRule(
                new_pattern,
                self._replacement_pattern,
                self._condition_function,
                matcher_class(new_pattern),
                self._verbose,
                self.name,
                self.remove_nodes,
                self.graph_pre_visitor,
                self.graph_post_visitor,
                self.as_function,
            )

        return [replace_pattern(p) for p in self._target_pattern.commute()]


class RewriteRuleClassBase(abc.ABC):
    """Base class for implementing rewrite rules as a class.

    Example::

        class TransposeIdentity(RewriteRuleClassBase):
            def pattern(cls, op, x, perm):
                return op.Transpose(x, perm=perm)

            def check(cls, context, x: ir.Value, perm: ir.Attr) -> bool:
                if perm.is_ref():
                    return False
                if perm.type == ir.AttributeType.INTS:
                    if perm.as_ints() == list(range(len(perm.as_ints()))):
                        return True
                return False

            def rewrite(cls, op, x: ir.Value, perm: ir.Attr | None = None):
                return op.Identity(x)

        # Then use
        # TransposeIdentity.rule()
        # to create a RewriteRule object.

    """

    @classmethod
    def rule(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        return RewriteRule(
            instance.pattern,
            instance.rewrite,
            instance.check,
            name=instance.name,
            remove_nodes=instance.remove_nodes,
            graph_pre_visitor=instance.setup,
            graph_post_visitor=instance.cleanup,
            as_function=instance.as_function,
        )

    def __init__(
        self, name: str | None = None, remove_nodes: bool = True, as_function: bool = False
    ) -> None:
        self.name = name or self.__class__.__name__
        self.remove_nodes = remove_nodes
        self.as_function = as_function

    @abc.abstractmethod
    def pattern(self, op, *args, **kwargs):
        raise NotImplementedError("Method 'pattern' must be implemented by derived class.")

    def check(self, op, *args, **kwargs) -> _basics.MatchResult:
        """Default check function that returns a _basics.MatchResult object with success always set to True."""
        return _basics.MatchResult()

    @abc.abstractmethod
    def rewrite(self, op, *args, **kwargs):
        raise NotImplementedError("Method 'rewrite' must be implemented by derived class.")

    def setup(self):
        """Optional setup function that can be overridden by derived classes.

        Used to do per model/function initialization.
        """
        return

    def cleanup(self):
        """Optional cleanup function that can be overridden by derived classes.

        Used to do per model/function cleanup.
        """
        return


def _copy_for_function(
    inputs: Sequence[ir.Value | None], nodes: Sequence[ir.Node], outputs: Sequence[ir.Value]
):
    """Utility function to extract a subgraph out as a function."""
    value_map: dict[ir.Value, ir.Value] = {}
    function_inputs: list[ir.Value] = []
    constant_nodes: list[ir.Node] = []
    for input in inputs:
        # Create a function input (formal-parameter value) to represent this value:
        new_value = (
            ir.Value(
                name=input.name,
                shape=input.shape,
                type=input.type,
                doc_string=input.doc_string,
            )
            if input
            else ir.Value()  # dummy parameter for a None input
        )
        if input is not None:
            value_map[input] = new_value
        function_inputs.append(new_value)

    def copy_value(value: ir.Value | None) -> ir.Value | None:
        if value is None:
            return None
        if value not in value_map:
            const_value = value.const_value
            if const_value is not None:
                # create a Constant node to represent the value
                value_attr = ir.AttrTensor("value", const_value)
                const_node = ir.Node("", "Constant", [], [value_attr])
                constant_nodes.append(const_node)
                value_map[value] = result = const_node.outputs[0]
                return result
            raise ValueError(f"Value {value} not found in value_map.")
        return value_map[value]

    def copy_attr_value(attr: ir.Attr) -> ir.Attr:
        if attr.is_ref():
            # No need to support this currently, as rewriting inside a function is
            # not used, as it has several challenges.
            raise NotImplementedError("RefAttr not supported.")
        if attr.type in {ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS}:
            # No need to support this currently, as rewriting control-flow constructs
            # is not used and has several challenges.
            raise NotImplementedError("Graph attributes not supported.")
        # Primitive attributes are immutable by design and can be shared.
        return attr

    def copy_node(node: ir.Node) -> ir.Node:
        new_inputs = [copy_value(v) for v in node.inputs]
        new_attributes = [copy_attr_value(v) for v in node.attributes.values()]
        new_node = ir.Node(
            node.domain,
            node.op_type,
            new_inputs,
            new_attributes,
            overload=node.overload,
            num_outputs=len(node.outputs),
            graph=None,
            name=node.name,
            doc_string=node.doc_string,  # type: ignore
            metadata_props=node.metadata_props.copy(),
        )
        new_outputs = new_node.outputs
        for i, output in enumerate(node.outputs):
            value_map[output] = new_outputs[i]
            if output.name is not None:
                new_outputs[i].name = output.name
        return new_node

    function_nodes = [copy_node(node) for node in nodes]
    function_outputs = [copy_value(v) for v in outputs]
    return (function_inputs, constant_nodes + function_nodes, function_outputs)


def _get_new_overload(model: ir.Model, domain: str, name: str) -> str:
    """Get a new overload for the given domain and name.

    Args:
        model: The model to which the new overload will be added.
        domain: The domain of the new overload.
        name: The opname of the new overload.

    Returns:
        The new overload name.
    """
    existing_functions = model.functions
    # Just a simple implementation for now
    overload = 1
    while True:
        overload_name = str(overload)
        if (domain, name, overload_name) not in existing_functions:
            return overload_name
        overload += 1


class RewriteRuleSet:
    def __init__(self, rules: Sequence[RewriteRule], *, commute: bool = False) -> None:
        if not rules:
            raise ValueError("rules must contain at least one rule")
        if commute:
            rules = list(itertools.chain.from_iterable([rule.commute() for rule in rules]))
        self.rules = rules
        # We call remove_unused_nodes at end of rewriting if there is any rule that does
        # NOT remove nodes (immediately when it is applied)
        self.remove_unused_nodes = any(not rule.remove_nodes for rule in rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rules})"

    def _apply_to_graph_or_function(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        *,
        verbose: int | None,
        tracer: _basics.MatchingTracer | None = None,
    ) -> int:
        """
        Apply the rewrite rules to the given graph or function.

        Args:
            model: The model to which the rewrite rules are applied.
            graph_or_function: The graph or function to which the rewrite rules are applied.
            verbose: The verbosity level. Defaults to None.
            tracer: The tracer for debugging. Defaults to None.

        Returns:
            The number of rewrite rules applied.
        """
        count = 0

        # NOTE: Rules should be prioritized in the order they are added to the RewriteRuleSet.
        # And the graph is applied in order.
        for rule in self.rules:
            if rule.graph_pre_visitor:
                rule.graph_pre_visitor()
            for node in graph_or_function:
                delta = rule.try_rewrite(
                    model, graph_or_function, node, verbose=verbose, tracer=tracer
                )
                if delta is None or tracer is not None:
                    continue
                assert isinstance(delta, ReplacementSubgraph)
                if delta.new_initializers:
                    if isinstance(graph_or_function, ir.Function):
                        # TODO(rama): Can't add initializers to functions. But currently this is not
                        # an issue, as we apply inlining before applying rewrite rules.
                        if verbose:
                            print(
                                f"Rewrites adding initializers not supported for functions: {rule}"
                            )
                        continue
                    initializers = graph_or_function.initializers
                    for initializer in delta.new_initializers:
                        if initializer.name in initializers:
                            if verbose:
                                print(f"Initializer {initializer.name} already exists.")
                            continue
                    for initializer in delta.new_initializers:
                        initializers[initializer.name] = initializer  # type: ignore[index]
                # TODO: This does not yet handle the problem of determining the correct insertion point
                # for inserted nodes in the case of patterns with multiple output-nodes. The following
                # is sufficient for patterns with a single output-node "node", which can serve as the
                # insertion-point.
                onnxscript.optimizer.basic_constant_propagation(delta.new_nodes)
                if rule.as_function:
                    # Create a function out of a copy of the matched nodes
                    if len(delta.new_nodes) != 1:
                        raise ValueError(
                            "as_function=True is only supported for patterns with a single replacement node."
                        )
                    call_node = delta.new_nodes[0]
                    domain = call_node.domain
                    name = call_node.op_type
                    overload = _get_new_overload(model, domain, name)
                    call_node.overload = overload

                    # Create topologically sorted list of nodes to be replaced.
                    unsorted_nodes = set(delta.match.nodes)
                    original_nodes = [n for n in graph_or_function if n in unsorted_nodes]
                    # Create new inputs/nodes/outputs for the function
                    inputs, nodes, outputs = _copy_for_function(
                        call_node.inputs, original_nodes, delta.match.outputs
                    )

                    used_domains: set[str] = {node.domain for node in original_nodes}
                    parent_opset_imports = graph_or_function.opset_imports
                    used_opset_imports = {
                        k: v for k, v in parent_opset_imports.items() if k in used_domains
                    }

                    graph = ir.Graph(
                        inputs, outputs, nodes=nodes, opset_imports=used_opset_imports
                    )
                    f = ir.Function(domain, name, overload, graph=graph, attributes=())
                    model.functions[f.identifier()] = f
                convenience.replace_nodes_and_values(
                    graph_or_function,
                    node,
                    delta.match.nodes if rule.remove_nodes else [],
                    delta.new_nodes,
                    delta.match.outputs,
                    delta.new_outputs,
                )

                count += 1
            if rule.graph_post_visitor:
                rule.graph_post_visitor()

        return count

    def apply_to_model(
        self,
        model: ir.Model,
        *,
        verbose: int | None = None,
        tracer: _basics.MatchingTracer | None = None,
    ) -> int:
        """Apply the rewrite rules in the set to the model.

        Args:
            model: The model to which the rewrite rules are applied.
            verbose: The verbosity level of messages. Defaults to None.
            tracer: if specified, no changes are made to the model, only
                information about the best matches found is computed.

        Returns:
            The number of applications of rewrite rules.
        """
        assert isinstance(model, ir.Model)
        onnxscript.optimizer.basic_constant_propagation(model.graph)
        # Rewriting may introduce new functions. In the following loop,
        # we restrict rewriting to original functions, not newly introduced ones.
        original_functions = list(model.functions.values())
        count = self._apply_to_graph_or_function(
            model, model.graph, verbose=verbose, tracer=tracer
        )
        for function in original_functions:
            onnxscript.optimizer.basic_constant_propagation(function)
            count += self._apply_to_graph_or_function(
                model, function, verbose=verbose, tracer=tracer
            )
        if self.remove_unused_nodes:
            onnxscript.optimizer.remove_unused_nodes(model)
        return count

    def __iter__(self):
        yield from self.rules
