import onnxscript
import onnxscript.rewriter.generic_pattern as orgp
import onnxscript.rewriter.pattern as orp
from onnxscript import ir


def rule_multiply_by_one(verbose: int = 0) -> orp.RewriteRule:
    """Replaces Mul(X, 1) by Identity(X).

    Args:
        verbose: verbosity

    Returns:
        RewriteRule
    """
    op = onnxscript.opset18

    def pattern(x, cst):
        return op.Mul(x, cst)

    def validate(model: ir.Model, match_result: orgp.PatternMatchResult) -> bool:
        model_nodes = match_result.model_nodes
        assert len(model_nodes) == 1, f"Only one node is expected not {len(model_nodes)}."
        cst = model_nodes[0].inputs[1]
        name = cst.name

        # TODO: implement a method to check that a value is a constant, a scalar constant
        # TODO: implement a method to retrieve its value
        if name not in model.graph.initializers:
            return False
        value = model.graph.initializers[name]
        if value.shape not in ((1,), tuple()):
            return False

        # TODO: check this is cached, does not work work with bfloat16
        np_value = value.numpy()
        scalar = float(np_value[0] if len(np_value.shape) == 1 else np_value)
        return scalar == 1

    def apply(x, cst):
        return op.Identity(x)

    return orgp.make_pattern_rule(pattern, apply, validate)
