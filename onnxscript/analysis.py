import ast

from numpy import isin

def used_vars(expr):
    ''' Return set of all variables used with an expression.'''
    if (isinstance(expr, ast.Name)):
        return set([expr.id])
    result = set()
    for c in ast.iter_child_nodes(expr):
        result = result | used_vars(c)
    return result

def local_defs(lhs):
    '''Utility function to return set of assigned/defined variables in the lhs of an assignment statement.'''
    def get_id(e):
        assert isinstance(e, ast.Name), "Only simple assignments supported."
        return e.id
    if (isinstance(lhs, ast.Tuple)):
        return set([get_id(x) for x in lhs.elts])
    else:
        return set([get_id(lhs)])

def defs(stmt):
    '''
    Return the set of all variables that may be defined (assigned to) in an
    execution of input stmt.
    '''
    def block_defs(block):
        result = set()
        for s in block:
            result = result | defs(s)
        return result
    if (isinstance(stmt, ast.Assign)):
        return local_defs(stmt.targets[0])
    elif (isinstance(stmt, ast.Return)):
        return set()
    elif (isinstance(stmt, ast.If)):
        return block_defs(stmt.body) | block_defs(stmt.orelse)
    elif isinstance(stmt, list):
        return block_defs(stmt)
    else:
        raise ValueError("Unsupported statement type: " + type(stmt).__name__) 


def do_liveness_analysis(fun):
    '''
    Perform liveness analysis of the given function-ast. The results of the
    analysis are stored directly with each statement-ast `s` as attributes `s.live_in`
    and `s.live_out`. 
    '''
    def visit (stmt, live_out):
        def visitBlock(block, live_out):
            for s in reversed(block):
                live_out = visit(s, live_out)
            return live_out
        if (isinstance(stmt, ast.Assign)):
            return (live_out.difference(local_defs(stmt.targets[0]))) | used_vars(stmt.value)
        elif (isinstance(stmt, ast.Return)):
            return used_vars(stmt.value)
        elif (isinstance(stmt, ast.If)):
            live1 = visitBlock(stmt.body, live_out)
            live2 = visitBlock(stmt.orelse, live_out)
            return (live1 | live2) | used_vars(stmt.test)
        elif isinstance(stmt, ast.For):
            return live_out # TODO
        else:
            raise ValueError("Unsupported statement type: " + type(stmt).__name__)        
    assert type(fun) == ast.FunctionDef
    live = set()
    for s in reversed(fun.body):
        s.live_out = live
        live = visit(s, live)
        s.live_in = live
        # print(ast.dump(s))
        # print("Live-In = ", live)

def exposed_uses(stmts):
    '''
    Return the set of variables that are used before being defined by given block.
    '''
    def visitBlock(block, live_out):
        for stmt in reversed(block):
            live_out = visit(stmt, live_out)
        return live_out

    def visit (stmt, live_out):
        if (isinstance(stmt, ast.Assign)):
            return (live_out.difference(local_defs(stmt.targets[0]))) | used_vars(stmt.value)
        elif (isinstance(stmt, ast.Return)):
            return used_vars(stmt.value)
        elif (isinstance(stmt, ast.If)):
            live1 = visitBlock(stmt.body, live_out)
            live2 = visitBlock(stmt.orelse, live_out)
            return (live1 | live2) | used_vars(stmt.test)
        else:
            raise ValueError("Unsupported statement type: " + type(stmt).__name__)

    return visitBlock(stmts, set())