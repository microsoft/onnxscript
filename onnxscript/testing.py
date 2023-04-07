# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Public utilities for testing onnxscript."""

import onnx
from onnx import parser

import onnxscript

__all__ = ["assert_isomorphic", "assert_isomorphic_graph", "assert_isomorphic_function"]


def assert_isomorphic(graph_or_function_1, graph_or_function_2):
    """Assert two graphs or functions are isomorphic."""
    assert _isomorphic(
        _to_function_or_graph(graph_or_function_1),
        _to_function_or_graph(graph_or_function_2),
    )


def assert_isomorphic_graph(graph1, graph2):
    """Assert two graphs are isomorphic."""
    assert _isomorphic(_to_graph_proto(graph1), _to_graph_proto(graph2))


def assert_isomorphic_function(fn1, fn2):
    """Assert two functions are isomorphic."""
    assert _isomorphic(_to_function_proto(fn1), _to_function_proto(fn2))


def _default_equality_op(x, y):
    return x == y


def _same_optional(field, obj1, obj2, equals=_default_equality_op):
    """Check two proto object have same value for optional field.
    This is restricted to simple field types where == comparison is sufficient.
    """
    if obj1.HasField(field):
        return obj2.HasField(field) and equals(getattr(obj1, field), getattr(obj2, field))
    return not obj2.HasField(field)


def _same_repeated(values1, values2, equals=_default_equality_op):
    if len(values1) != len(values2):
        return False
    for val1, val2 in zip(values1, values2):
        if not equals(val1, val2):
            return False
    return True


def _same_string_string_map(proto1, proto2):
    """Compare repeated StringStringEntryProto as maps."""

    def to_map(proto):
        return {x.key: x.value for x in proto}

    return to_map(proto1) == to_map(proto2)


def _same_tensor(tp1, tp2):
    if tp1.dims != tp2.dims:
        return False
    if not _same_optional("data_type", tp1, tp2):
        return False
    # Segmented representation not supported yet
    if tp1.HasField("segment") or tp2.HasField("segment"):
        return False
    if tp1.float_data != tp2.float_data:
        return False
    if tp1.int32_data != tp2.int32_data:
        return False
    if tp1.string_data != tp2.string_data:
        return False
    if tp1.int64_data != tp2.int64_data:
        return False
    if tp1.uint64_data != tp2.uint64_data:
        return False
    if tp1.double_data != tp2.double_data:
        return False
    # Ignore name for comparison:
    # if not _same_optional("name", tp1, tp2): return False
    if not _same_optional("doc_string", tp1, tp2):
        return False
    if not _same_optional("data_location", tp1, tp2):
        return False
    if not _same_string_string_map(tp1.external_data, tp2.external_data):
        return False
    return True


def _same_dim(dim1, dim2):
    return _same_optional("dim_value", dim1, dim2) and _same_optional("dim_param", dim1, dim2)


def _same_shape(shape1, shape2):
    return _same_repeated(shape1.dim, shape2.dim, _same_dim)


def _same_tensor_type(tt1, tt2):
    return (tt1.elem_type == tt2.elem_type) and _same_optional("shape", tt1, tt2, _same_shape)


def _same_type(tp1, tp2):
    # Handles only tensor type at this point.
    return _same_optional("tensor_type", tp1, tp2, _same_tensor_type)


def _same_value_info(vi1, vi2):
    return (
        _same_optional("name", vi1, vi2)
        and _same_optional("type", vi1, vi2, _same_type)
        and _same_optional("doc_string", vi1, vi2)
    )


def _same_attr(attr1, attr2, graph_equality):
    # no name check; names used to match attributes already.
    for field in ["type", "ref_attr_name", "f", "i", "s"]:
        if not _same_optional(field, attr1, attr2):
            return False

    if not _same_optional("t", attr1, attr2, _same_tensor):
        return False

    if not _same_repeated(attr1.tensors, attr2.tensors, _same_tensor):
        return False

    for field in ["floats", "ints", "strings"]:
        if getattr(attr1, field) != getattr(attr2, field):
            return False

    if not _same_optional("g", attr1, attr2, graph_equality):
        return False

    if not _same_repeated(attr1.graphs, attr2.graphs, graph_equality):
        return False

    for field in ["sparse_tensor", "tp"]:
        # TODO(gramalingam): check for more complex fields
        if attr1.HasField(field) or attr2.HasField(field):
            return False
    return True


def _same_attrs(attrs1, attrs2, graph_equality):
    if len(attrs1) != len(attrs2):
        return False
    attrs1map = {a.name: a for a in attrs1}
    for attr2 in attrs2:
        if attr2.name not in attrs1map:
            return False
        attr1 = attrs1map[attr2.name]
        if not _same_attr(attr1, attr2, graph_equality):
            return False
    return True


def _ioname(x):
    """Return the name of an input/output of a function or graph"""
    return x.name if isinstance(x, onnx.ValueInfoProto) else x


class _Matcher:
    """An isomorphism matcher for two functions or two graphs."""

    def __init__(self, fg1, fg2, outer_scope) -> None:
        def defmap(f):
            """Compute a map from variables v to their definition-sites.
            A definition-site (n, i) indicates the i-th output of n-th node
            The special value (-1, i) is used to indicate the i-th input of a function/graph.
            """
            result = {}
            for i, x in enumerate(f.input):
                result[_ioname(x)] = (-1, i)
            for ni, n in enumerate(f.node):
                for xi, x in enumerate(n.output):
                    result[x] = (ni, xi)
            return result

        self.defmap1 = defmap(fg1)
        self.defmap2 = defmap(fg2)
        self.fg1 = fg1
        self.fg2 = fg2
        self.node_mapping: dict[onnx.NodeProto, onnx.NodeProto] = {}
        self.outer_scope = outer_scope

    def same_value(self, var1, var2):
        """Match two variables (strings)."""
        if var1 not in self.defmap1 or var2 not in self.defmap2:
            # If one of the variables is in current scope, or if there is no outer scope, fail
            if (var1 in self.defmap1) or (var2 in self.defmap2) or (self.outer_scope is None):
                return False
            # Both variables are in outer-scopes. Delay check until later
            return self.outer_scope.same_value(var1, var2)
        (node1, index1) = self.defmap1[var1]
        (node2, index2) = self.defmap2[var2]
        return (index1 == index2) and self.same_node(node1, node2)

    def same_node(self, n1, n2):
        """Match two node-indices. The special node-index -1 represents inputs."""
        if (n1 == -1) and (n2 == -1):
            return True  # Both are inputs
        if (n1 == -1) or (n2 == -1):
            return False  # Only one is input
        if n1 in self.node_mapping:
            return self.node_mapping[n1] == n2
        node1 = self.fg1.node[n1]
        node2 = self.fg2.node[n2]
        if node1.op_type != node2.op_type:
            return False
        if node1.domain != node2.domain:
            return False
        # check attrs
        if not _same_attrs(node1.attribute, node2.attribute, self.same_sub_graph):
            return False
        if not self.same_value_list(node1.input, node2.input):
            return False

        # Nodes represent same computation. Cache the comparison result.
        self.node_mapping[n1] = n2
        return True

    def same_value_list(self, list1, list2):
        """Match two lists of variables (either a string or ValueInfoProto)"""
        if len(list1) != len(list2):
            return False
        for x, y in zip(list1, list2):
            if not self.same_value(_ioname(x), _ioname(y)):
                return False
        return True

    def same_sub_graph(self, g1, g2):
        """Match two sub-graphs."""
        sub_graph_matcher = _Matcher(g1, g2, self)
        return sub_graph_matcher.same_graph()

    def same_graph(self):
        """Match two sub-graphs."""
        g1 = self.fg1
        g2 = self.fg2
        if not _same_repeated(g1.input, g2.input, _same_value_info):
            return False

        if g1.initializer or g2.initializer:
            return False  # TODO
        if g1.sparse_initializer or g2.sparse_initializer:
            return False  # TODO
        if not self.same_value_list(g1.output, g2.output):
            return False
        # TODO completeness tests!
        return True

    def same_function(self):
        """Match (top-level) two functions."""

        # Ok for function names/domain to be different.

        if len(self.fg1.input) != len(self.fg2.input):
            return False
        if set(self.fg1.attribute) != set(self.fg2.attribute):
            return False

        # Opset imports must be same (but possibly in different order):
        # Convert opset-imports into a dictionary
        def imports(f):
            # Assumes each domain has only one entry in a valid FunctionProto
            return {entry.domain: entry.version for entry in f.opset_import}

        if imports(self.fg1) != imports(self.fg2):
            return False

        # Now do a specific form of isomorphism check: Both must compute the same
        # set of operations, possibly in different order as long as they respect
        # the topological-sort order requirement. The two may use different names
        # for intermediate-values, as long as the computation is the same.

        if len(self.fg1.node) != len(self.fg2.node):
            return False

        if not self.same_value_list(self.fg1.output, self.fg2.output):
            return False

        # We do not allow for unused values in the function, which are
        # hard to handle in an isomorphism check.
        if len(self.node_mapping) != len(self.fg1.node):
            return False
        if len(set(self.node_mapping.values())) != len(self.fg2.node):
            return False

        return True


def _isomorphic(fg1, fg2):
    """Checks that two function/graph bodies are isomorphic.
    Assumes that the inputs are valid FunctionProto/GraphProto.
    Use a separate check to verify that the inputs satisfy
    FunctionProto/GraphProto requirements (like no duplicate attributes).
    """
    matcher = _Matcher(fg1, fg2, None)
    if isinstance(fg1, onnx.FunctionProto):
        if not isinstance(fg2, onnx.FunctionProto):
            raise TypeError("Both inputs must be same type (function or graph)")
        return matcher.same_function()
    if isinstance(fg1, onnx.GraphProto):
        if not isinstance(fg2, onnx.GraphProto):
            raise TypeError("Both inputs must be same type (function or graph)")
        return matcher.same_graph()
    raise TypeError("Inputs must be either a FunctionProto or GraphProto")


def _to_function_proto(f):
    if isinstance(f, onnx.FunctionProto):
        return f
    if isinstance(f, onnxscript.OnnxFunction):
        return f.to_function_proto()
    if isinstance(f, str):
        return parser.parse_function(f)
    raise TypeError(f"Cannot convert {type(f)} to FunctionProto")


def _to_graph_proto(g):
    if isinstance(g, onnx.GraphProto):
        return g
    if isinstance(g, onnxscript.OnnxFunction):
        return g.to_model_proto().graph
    if isinstance(g, str):
        return parser.parse_graph(g)
    raise TypeError(f"Cannot convert {type(g)} to ModelProto")


def _to_function_or_graph(obj):
    if isinstance(obj, onnx.FunctionProto):
        return obj
    if isinstance(obj, onnx.GraphProto):
        return obj
    if isinstance(obj, onnx.ModelProto):
        return obj.graph
    if isinstance(obj, onnxscript.OnnxFunction):
        return obj.to_function_proto()
    raise TypeError(f"Cannot convert {type(obj)} to FunctionProto or GraphProto")
