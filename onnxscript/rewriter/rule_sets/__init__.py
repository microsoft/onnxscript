import onnxscript.rewriter.pattern as orp
import onnxscript.rewriter.rule_sets.llm_generic_rules_p0 as p0


# starts matching
def llm_generic_rule_set_p0(verbose: int = 0) -> orp.RewriteRuleSet:
    """Returns a set of rules which should be applied
    before anyother one as they usually remove unnecessary computation
    such as the multiplication by 1.

    Args:
        verbose: verbosity

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet([p0.rule_multiply_by_one(verbose=verbose)])
