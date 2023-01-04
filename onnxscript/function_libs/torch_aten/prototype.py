
def run_symbolic_function(name, *args):
    symbolic_remainder = lookup(name)

    # Convert args into SymbolicTensors
    symbolic_args = [SymbolicTensor(arg) for arg in args]

    # Validate the input to make sure it has enough data to be dispatched
    check_signature(symbolic_remainder, symbolic_args)

    # looks up the function and calls, e.g.
    symbolic_remainder(*args)


@symbolic_function("aten::remainder")
@spec(
    {
        "a": [Policy.RequiresDType]
        "b": [Policy.RequiresDType]
    }
)
def symbolic_remainder(a: SymbolicTensor, b: SymbolicTensor) -> SymbolicTensor:
    # compare dtype
    if check_signature(aten_remainder_int):
        return aten_remainder_int(x)

    a, b = promote_types(a, b)
    # dispatch to the default function
    return aten_remainder(a, b)


def promote_types(a, b):
    a = op.Cast(a, dtype)
    b = op.Cast(b, dtype)
    return a, b


class TracingEvaluator(Evaluator):
    def _eval():
        # record the graph
