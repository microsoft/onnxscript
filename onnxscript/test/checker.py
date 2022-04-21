from platform import node
import onnx


def same_optional(field, obj1, obj2, equals=None):
    '''
    Check two proto object have same value for optional field.
    This is restricted to simple field types where == comparison is sufficient.
    '''
    if (equals is None):
        equals = lambda v1, v2: v1 == v2
    if (obj1.HasField(field)):
        return obj2.HasField(field) and equals(getattr(obj1, field), getattr(obj2, field))
    else:
        return not obj2.HasField(field)
    

def same_attr(attr1, attr2, graph_equality):
    # no name check; names used to match attributes already.
    for field in ["type", "ref_attr_name", "f", "i", "s"]:
        if not same_optional(field, attr1, attr2):
            return False

    for field in ["floats", "ints", "strings"]:
        if getattr(attr1, field) != getattr(attr2, field):
            return False

    if not same_optional("g", attr1, attr2, graph_equality):
        return False

    for (g1, g2) in zip (attr1.graphs, attr2.graphs):
        if not graph_equality(g1, g2):
            return False

    # for field in ["t", "sparse_tensor", "tp", "tensors", "sparse_tensors", "type_protos"]:
    for field in ["t", "sparse_tensor", "tp"]:
        # TODO: check for more complex fields
        if attr1.HasField(field) or attr2.HasField(field):
            return False
    return True


def same_attrs(attrs1, attrs2, graph_equality):
    if len(attrs1) != len(attrs2):
        return False
    attrs1map = {a.name: a for a in attrs1}
    for attr2 in attrs2:
        if attr2.name not in attrs1map:
            return False
        attr1 = attrs1map[attr2.name]
        if not same_attr(attr1, attr2, graph_equality):
            return False
    return True

# Return the name of an input/output of a function or graph
def ioname(x):
    return x.name if isinstance(x, onnx.ValueInfoProto) else x

def isomorphic(fn1: onnx.FunctionProto, fn2: onnx.FunctionProto):
    '''
    Checks that two function bodies are isomorphic.
    Assumes that the inputs are valid FunctionProto.
    Use a separate check to verify that the inputs satisfy
    FunctionProto requirements (like no duplicate attributes).
    '''
    # Ok for function names/domain to be different.

    # Attribute parameters and inputs must be same for both:
    if (fn1.input != fn2.input): return False
    if (fn1.attribute != fn2.attribute): return False

    # Opset imports must be same (but possibly in different order):

    # Convert opset-imports into a dictionary
    def imports(f):
        # TODO: assuming each domain has only one entry in a valid FunctionProto
        return {entry.domain: entry.version for entry in f.opset_import}

    if (imports(fn1) != imports(fn2)): return False

    # Now do a specific form of isomorphism check: Both must compute the same
    # set of operations, possibly in different order as long as they respect
    # the topological-sort order requirement. The two may use different names
    # for intermediate-values, as long as the computation is the same.

    if len(fn1.node) != len(fn2.node): return False

    # Compute a map from variables v to their definition-sites.
    # A definition-site (n, i) indicates the i-th output of n-th node
    # The special value (-1, i) is used to indicate the i-th input of a function/graph
    def defmap(f):
        result = {}
        for (i, x) in enumerate(f.input):
            result[ioname(x)] = (-1, i)
        for ni, n in enumerate(f.node):
            for xi, x in enumerate(n.output):
                result[x] = (ni, xi) 
        return result

    class Scope:
        def __init__(self, fg1, fg2, outer_scope) -> None:
            self.fn1map = defmap(fg1)
            self.fn2map = defmap(fg2)
            self.nodelist1 = fg1.node
            self.nodelist2 = fg2.node
            self.node_mapping = {}
            self.delayed_checks = []
            self.outer_scope = outer_scope

    scope = Scope(fn1, fn2, None)

    # Check that fn1 computes the same value for var1 as fn2 computes for var2:
    def same_value(var1, var2):
        if (var1 not in scope.fn1map or var2 not in scope.fn2map):
            # If one of the variables is in current scope, or if there is no outer scope, fail
            if (var1 in scope.fn1map) or (var2 in scope.fn2map) or (scope.outer_scope is None): return False
            # For variables in outer-scopes, delay check until later
            scope.delayed_checks.append((var1, var2))
            return True
        (node1, index1) = scope.fn1map[var1]
        (node2, index2) = scope.fn2map[var2]
        return (index1 == index2) and same_node(node1, node2)

    def same_node(n1, n2):
        if (n1 == -1) and (n2 == -1): return True # Both are inputs
        if (n1 == -1) or (n2 == -1): return False # Only one is input
        if (n1 in scope.node_mapping):
            return scope.node_mapping[n1] == n2
        node1 = scope.nodelist1[n1]
        node2 = scope.nodelist2[n2]
        if node1.op_type != node2.op_type: return False
        if node1.domain != node2.domain: return False
        # check attrs
        if not same_attrs(node1.attribute, node2.attribute, same_sub_graph): return False
        if not same_outputs(node1.input, node2.input): return False # TODO: fix name 'same_outputs'

        # Nodes represent same computation. Cache the comparison result.
        scope.node_mapping[n1] = n2
        return True

    def same_sub_graph(g1, g2):
        if len(g1.input) != len(g2.input): return False
        # TODO: check types
        if g1.initializer or g2.initializer: return False # TODO
        if g1.sparse_initializer or g2.sparse_initializer: return False # TODO

        # Save information about outer scope TODO: delayed_checks?
        nonlocal scope
        scope = Scope(g1, g2, scope)

        if not same_outputs(g1.output, g2.output): return False
        # TODO completeness tests!
     
        # Restore information about outer scope
        outer_scope_checks = scope.delayed_checks
        scope = scope.outer_scope

        for (v1, v2) in outer_scope_checks:
            if not same_value(v1, v2): return False
        return True
    
    def same_outputs(list1, list2):
        # Check that both functions compute the same value for all outputs:
        if len(list1) != len(list2): return False

        # For now, we allow the function outputs to have different names.
        for x, y in zip(list1, list2):
            if not same_value(ioname(x), ioname(y)):
                return False
        
        return True

    if not same_outputs(fn1.output, fn2.output): return False

    # We do not allow for unused values in the function, which are
    # hard to handle in an isomorphism check.
    if len(scope.node_mapping) != len(fn1.node): return False
    if len(set(scope.node_mapping.values())) != len(fn2.node): return False

    return True
