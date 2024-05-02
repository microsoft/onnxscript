import onnxscript.rewriter.no_op as no_op
import onnxscript.rewriter.pattern as orp
from onnxscript.rewriter import pattern

op = pattern.onnxop


def transpose_identity(x):
    return op.Transpose(x, perm=[0, 1, 2])


def transpose_transpose(x):
    return op.Transpose(op.Transpose(x, perm=[1, 0]), perm=[1, 0])


# def same_transpose(x):
#    return op.Transpose(x, perm=[3,2,1,0]), op.Transpose(x, perm=[3,2,1,0])

transpose_identity_rule = pattern.RewriteRule(transpose_identity, no_op.identity)
transpose_transpose_rule = pattern.RewriteRule(transpose_transpose, no_op.identity)


def llama_p0_rule_set(verbose: int = 0) -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before anyother one as they usually remove unnecessary computation
    such as the multiplication by 1.

    Args:
        verbose: verbosity
    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            no_op.mul_by_1_rule,
            no_op.add_0_rule,
            no_op.add_0_rule,
            no_op.div_by_1_rule,
            transpose_identity_rule,
            transpose_transpose_rule,
        ]
    )
