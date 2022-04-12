import onnx


def isomorphic(fn1: onnx.FunctionProto, fn2: onnx.FunctionProto):
    '''
    Checks that two function bodies are isomorphic.
    '''
    # Ok for function names/domain to be different.

    # Attributes, inputs, and outputs must be same for both:
    if (fn1.input != fn2.input):
        return False
    if (fn1.output != fn2.output):
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

        # Nodes represent same computation. Cache the comparison result.
        node_mapping[n1] = n2
        return True

    # Check that both functions compute the same value for all outputs:
    for x in fn1.output:
        if not same_value(x, x):
            return False

    # TODO: This doesn't yet handle/check for unused computed values.
    return True
