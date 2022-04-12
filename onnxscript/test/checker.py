import onnx

def same_optional(field, obj1, obj2):
    '''
    Check two proto object have same value for optional field.
    This is restricted to simple field types where == comparison is sufficient.
    '''
    if (obj1.HasField(field)):
        return obj2.HasField(field) and (getattr(obj1, field) == getattr(obj2, field))
    else:
        return not obj2.HasField(field)

def isomorphic(fn1: onnx.FunctionProto, fn2: onnx.FunctionProto):
    '''
    Checks that two function bodies are isomorphic.
    Assumes that the inputs are valid FunctionProto.
    Use a separate check to verify that the inputs satisfy
    FunctionProto requirements (like no duplicate attributes).
    '''
    # Ok for function names/domain to be different.

    # Attribute parameters and inputs must be same for both:
    if (fn1.input != fn2.input):
        return False
    if (fn1.attribute != fn2.attribute):
        return False

    # Opset imports must be same (but possibly in different order):

    # Convert opset-imports into a dictionary
    def imports(f):
        # TODO: assuming each domain has only one entry in a valid FunctionProto
        return {entry.domain: entry.version for entry in f.opset_import}

    if (imports(fn1) != imports(fn2)):
        return False

    # Now do a specific form of isomorphism check: Both must compute the same
    # set of operations, possibly in different order as long as they respect
    # the topological-sort order requirement. The two may use different names
    # for intermediate-values, as long as the computation is the same.

    if len(fn1.node) != len(fn2.node): return False

    def defmap(f):
        return {x: (ni, xi) for ni, n in enumerate(f.node)
                for xi, x in enumerate(n.output)}

    fn1map = defmap(fn1)
    fn2map = defmap(fn2)

    # Check that fn1 computes the same value for var1 as fn2 computes for var2:
    def same_value(var1, var2):
        # Inputs must match each other:
        if var1 in fn1.input or var2 in fn2.input:
            return var2 == var2
        if var2 in fn2.input:
            return False

        # Computed values must be computed by isomorphic nodes:
        if (var1 not in fn1map or var2 not in fn2map):
            return False  # Undefined variables

        (node1, index1) = fn1map[var1]
        (node2, index2) = fn2map[var2]

        return (index1 == index2) and same_node(node1, node2)

    node_mapping = {}

    def same_node(n1, n2):
        nonlocal node_mapping
        if (n1 in node_mapping):
            return node_mapping[n1] == n2
        node1 = fn1.node[n1]
        node2 = fn2.node[n2]
        if node1.op_type != node2.op_type:
            return False
        if node1.domain != node2.domain:
            return False
        # check attrs
        if len(node1.attribute) != len(node2.attribute): return False
        attrs1 = { a.name : a for a in node1.attribute }
        for attr2 in node1.attribute:
            if attr2.name not in attrs1: return False
            attr1 = attrs1[attr2.name]

            for field in ["type", "ref_attr_name", "f", "i", "s", "floats", "ints", "strings"]:
                if not same_optional(field, attr1, attr2): return False
            # TODO: for graph/tensor/other fields

        # Nodes represent same computation. Cache the comparison result.
        node_mapping[n1] = n2
        return True

    # Check that both functions compute the same value for all outputs:
    if len(fn1.output) != len(fn2.output): return False
    # For now, we allow the function outputs to have different names.
    for x, y in zip(fn1.output, fn2.output):
        if not same_value(x, y):
            return False

    # We do not allow for unused values in the function, which are
    # hard to handle in an isomorphism check.
    if len(node_mapping) != len (fn1.node): return False
    if len(set(node_mapping.values())) != len (fn2.node): return False

    return True
