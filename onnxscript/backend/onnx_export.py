# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy
import onnx
from onnx.helper import make_node
from onnx import numpy_helper, ModelProto
from ..onnx_types import ParametricTensor


_template_python = '''
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script
from onnxscript.values import Opset
from onnxscript.onnx_types import {{ ", ".join(unique_types) }}
from onnxscript.onnx_opset import opset{{ opsets[''] }}

{% for domain, version in unique_function_domain_version: %}
{{ domain }}{{ version }} = Opset("{{ domain }}", {{ version }}){% endfor %}

{% for domain, name, fct in functions: %}

@script({{ domain }}{{ version }})
def {{ python_make_node_name(fct['proto'].domain, 1, fct['proto'].name) }}({{
    ", ".join(map(rename, fct['proto'].input)) }}):
    # attributes are missing
    {% if fct['proto'].doc_string %}"""
    {{ fct['proto'].doc_string }}
    """{%- endif %}
    {%- for node in fct['proto'].node: %}
{{ python_make_node(node, opsets, indent=1) }}{% endfor %}
    return {{ ", ".join(map(rename, fct['proto'].output)) }}

{% endfor %}

@script()
def {{ function_name }}({% if graph.input: %}{{ rename(graph.input[0].name) }}: {{
    translate(graph.input[0].type) }}{% endif %}{% for i in graph.input[1:]:
%}, {{ rename(i.name) }}: {{ translate(i.type) }}{% endfor %}) -> ({{
    translate(graph.output[0].type) }}{% for o in graph.output[1:]: %}, {{
    translate(o.type) }}{% endfor %}):
    {% if doc_string %}"""
    {{ doc_string }}
    """{%- endif %}
{{ python_make_node_graph(graph, opsets, indent=1) }}
    return {{ rename(graph.output[0].name) }}{%
        for o in graph.output[1:]: %}, {{ rename(o.name) }}{% endfor %}
'''


kwlist = {
    'False',
    'None',
    'True',
    'and',
    'as',
    'assert',
    'async',
    'await',
    'break',
    'class',
    'continue',
    'def',
    'del',
    'elif',
    'else',
    'except',
    'finally',
    'for',
    'from',
    'global',
    'if',
    'import',
    'in',
    'is',
    'lambda',
    'nonlocal',
    'not',
    'or',
    'pass',
    'raise',
    'return',
    'try',
    'while',
    'with',
    'yield'
}


def _rename_variable(name):
    """
    Renames all names equal to a python keyword.
    """
    if name in kwlist:
        return 'r_' + name
    if name == "":
        return None
    return name


def _rename_variable_s(name):
    """
    Renames all names equal to a python keyword.
    """
    return str(_rename_variable(name))


def _translate_type(onnx_type):
    """
    Converts a onnx type into a type defined by *onnx-script*.
    """
    if onnx_type.HasField('tensor_type'):
        typ = ParametricTensor.types[onnx_type.tensor_type.elem_type]
        name = repr(typ)
        if onnx_type.tensor_type.HasField('shape'):
            shape = []
            for d in onnx_type.tensor_type.shape.dim:
                if d.HasField('dim_value'):
                    shape.append(str(d.dim_value))
                else:
                    shape.append(d.dim_param)
            if len(shape) == 0:
                return name
            return "%s[%s]" % (name, ",".join(shape))
        return name + "[...]"
    raise NotImplementedError(
        f"Unable to translate type {onnx_type!r} into onnx-script type.")


def _to_str(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s


def _attribute_value(attr):
    if attr.HasField("f"):
        return attr.f
    if attr.HasField("i"):
        return attr.i
    if attr.HasField("s"):
        return _to_str(attr.s)
    if attr.HasField("t"):
        return numpy_helper.to_array(attr.t)
    if attr.floats:
        return list(attr.floats)
    if attr.ints:
        return list(attr.ints)
    if attr.strings:
        return list(map(_to_str, attr.strings))
    raise NotImplementedError(f"Unable to return a value for attribute {attr!r}")


def _python_make_node_name(domain, version, name, node=False):
    if node:
        if version is None:
            version = 1
        if not isinstance(version, int):
            raise TypeError(
                "version must be an integer not %r for domain=%r and name=%r." % (
                    version, domain, name))
        if domain == '':
            return "opset%d.%s" % (version, name)
        return "%s%d.%s" % (domain.replace(".", "_"), version, name)
    return name


def _python_make_node_graph(graph, opsets, indent=0, output_names=None):
    """
    Translates a GraphProto into python.
    """
    code = []
    sindent = '    ' * indent
    if len(graph.initializer) > 0:
        code.append(f"{sindent}# initializer")
    for init in graph.initializer:
        node = make_node('Constant', [], [_rename_variable(init.name)], value=init)
        code.append(_python_make_node(node, opsets, indent=indent))
    if len(graph.sparse_initializer) > 0:
        if len(graph.initializer) > 0:
            code.append(f"{sindent}# sparse_initializer")
        raise NotImplementedError(
            "Unable to convert sparse_initilizer into python.")
    code.append(f"{sindent}# subgraph begins")
    for node in graph.node:
        code.append(_python_make_node(node, opsets, indent=indent))
    code.append(f"{sindent}# subgraph ends")
    if output_names is not None:
        code.append(f"{sindent}# output_names")
        for fr, to in zip(graph.output, output_names):
            code.append(
                "%s%s = %s" % (sindent, _rename_variable(to), _rename_variable(fr.name)))
    final = "\n".join(code)
    return final


def _python_make_node_make_attribute_str(node):
    attributes = []
    for at in node.attribute:
        value = _attribute_value(at)
        if isinstance(value, str):
            attributes.append((at.name, "%r" % value))
            continue
        if isinstance(value, numpy.ndarray):
            onnx_dtype = at.t.data_type
            if len(value.shape) == 0:
                text = (
                    'make_tensor("value", %s, dims=[], vals=[%r])'
                    '' % (onnx_dtype, value.tolist()))
            else:
                text = (
                    'make_tensor("value", %s, dims=%r, vals=%r)'
                    '' % (onnx_dtype, list(value.shape),
                          value.ravel().tolist()))
            attributes.append((at.name, text))
            continue
        attributes.append((at.name, repr(value)))

    return ", ".join("%s=%s" % (k, v) for k, v in attributes)


def _python_make_node_if(node, opsets, indent=0):
    """
    Translates a node If into python.
    """
    sindent = '    ' * indent
    code = ["%sif %s:" % (sindent, node.input[0])]
    if len(node.attribute) != 2:
        raise RuntimeError(
            f"Node {node.op_type!r} expected two attributes not {len(node.attribute)}.")
    atts = node.attribute
    if atts[0].name == 'else_branch':
        else_branch, then_branch = atts[0].g, atts[1].g
    else:
        else_branch, then_branch = atts[1].g, atts[0].g
    code.append(_python_make_node_graph(
        then_branch, opsets, indent=indent + 1,
        output_names=node.output))
    code.append("%selse:" % sindent)
    code.append(_python_make_node_graph(
        else_branch, opsets, indent=indent + 1,
        output_names=node.output))
    return "\n".join(code)


def _python_make_node_loop(node, opsets, indent=0):
    """
    Translates a node Loop into python.
    """
    body = node.attribute[0].g
    sindent = "    " * indent
    n_iter = _rename_variable(node.input[0])
    cond = _rename_variable(node.input[1])
    
    rows = []

    # node has 2 + N inputs and N + K outputs
    # body has 2 + N inputs and 1 + N + K outputs
    N = len(node.input) - 2
    K = len(node.output) - N
    assert len(node.input) == 2 + N
    assert len(node.output) == N + K
    assert len(body.input) == 2 + N
    assert len(body.input) == 1 + N + K
    for nin, gin in zip(node.input[2: 2 + N], body.input[2: 2 + N]):
        if gin.name != nin:
            rows.append(f"{sindent}{gin.name} = {nin}")
    for nout in node.output[N:]:
        rows.append(f"{sindent}{nout} = []")
    rows.append(f"{sindent}{body.input[1].name} = {node.input[1]}")
    
    if n_iter and not cond:
        rows.append("%sfor %s in range(%s):" % (
            sindent, body.input[0].name, n_iter))
    elif not n_iter and cond:
        rows.append("%sfor %s in conditional_range(None, %s):" % (sindent, cond))
    elif n_iter and cond:
        rows.append("%sfor %s in conditional_range(%s, %s):" % (
            sindent, body.input[0].name, n_iter, cond))
    else:
        raise RuntimeError(
            f"Unable to export loop type {node.op_type!r} into python because there is no "
            f"stop condition.")
    rows.append(_python_make_node_graph(body, opsets, indent=indent+1))

    # prepare for the next iteration
    for gin, gout in zip(body.input[2: 2 + N], body.output[1:1 + N]):
        if gin.name != gout.name:
            rows.append(f"{sindent}    {gin.name} = {gout.name}")
    for nout, gout in zip(node.output[N:], body.output[1 + N:]):
        rows.append(f"{sindent}    {nout}.append({gout.name})")

    # condition
    rows.append(f"{sindent}    {node.input[1]} &= {body.output[0].name}")
    rows.append(f"{sindent}    {body.input[1].name} = {body.output[0].name}")

    return "\n".join(rows)


def _python_make_node_scan(node, opsets, indent=0):
    """
    Translates a node Scan into python.
    """
    raise NotImplementedError()


def _python_make_node(onnx_node, opsets, indent=0):
    if isinstance(onnx_node, dict):
        node = onnx_node['onnx_node']
    else:
        node = onnx_node

    if node.op_type in {'If', 'Loop', 'Scan'}:
        # If, Loop, Scan
        if node.op_type == 'If':
            return _python_make_node_if(node, opsets, indent=indent)
        if node.op_type == 'Loop':
            return _python_make_node_loop(node, opsets, indent=indent)
        if node.op_type == 'Scan':
            return _python_make_node_scan(node, opsets, indent=indent)
        raise RuntimeError(
            "Unable to export node type %r into python." % (node.op_type, ))

    if any(map(lambda att: hasattr(att, 'g') and att.g and att.g.ByteSize() > 0,
               node.attribute)):
        raise RuntimeError(
            "Unable to export node type %r into python." % node.op_type)

    sindent = "    " * indent
    if node.op_type == 'Identity':
        rows = []
        for vin, vout in zip(node.input, node.output):
            rows.append(f"{sindent}{vout} = {vin}")
        return "\n".join(rows)

    ops = {'Add': '+', 'Sub': '-', 'Mul': '*', 'MatMul': '@',
           'Div': '/', 'Pow': '**',
           'And': '&', 'Or': '|', 'Greater': '>', 'Equal': '==',
           'Lesser': '<', 'GreaterOrEqual': '>=', 'LessOrEqual': '<='}
    if node.op_type in ops:
        return "%s%s = %s" % (sindent, _rename_variable(node.output[0]),
                              (" %s " % ops[node.op_type]).join(
                                map(_rename_variable, node.input)))
    name = _python_make_node_name(
        node.domain, opsets[node.domain], node.op_type, node=True)
    attributes_str = _python_make_node_make_attribute_str(node)
    if len(node.input) > 0 and len(attributes_str) > 0:
        attributes_str = ", " + attributes_str
    output_names = []
    for i, o in enumerate(node.output):
        if o in ('', None):
            output_names.append(f'_{i}')
        else:
            output_names.append(_rename_variable(o))

    text = [sindent, ", ".join(output_names), " = ", name,
            '(',
            ', '.join(map(_rename_variable_s, node.input)),
            attributes_str,
            ')']
    return "".join(text)


def export_template(model_onnx, template,
                    name=None, autopep_options=None,
                    function_name='main_function', clean_code=True):
    """
    Exports an ONNX model into a code based on a template.

    :param model_onnx: string or ONNX graph
    :param template: exporting template
    :param name: to overwrite onnx name
    :param autopep_options: :epkg:`autopep8` options
    :param function_name: main function name in the code
    :param clean_code: clean the code
    :return: python code
    """
    # delayed import to avoid raising an exception if not installed.
    import autopep8

    # unique_function_domain_version
    unique_function_domain_version = set()
    if hasattr(model_onnx, 'functions'):
        for f in model_onnx.functions:
            unique_function_domain_version.add((f.domain, 1))
    unique_function_domain_version = list(
        sorted(unique_function_domain_version))

    # containers
    context = {'main_model': model_onnx,
               'python_make_node': _python_make_node,
               'python_make_node_graph': _python_make_node_graph,
               'python_make_node_name': _python_make_node_name,
               'unique_function_domain_version': unique_function_domain_version,
               'rename': _rename_variable,
               'translate': _translate_type}

    # opset
    if hasattr(model_onnx, 'opset_import'):
        opsets = {}
        for oimp in model_onnx.opset_import:
            opsets[oimp.domain] = oimp.version
        context['opsets'] = opsets

    graph = model_onnx.graph if hasattr(model_onnx, 'graph') else model_onnx

    # types
    unique_types = set()
    for t in list(graph.input) + list(graph.output):
        ts = _translate_type(t.type)
        its = ts.split('[', maxsplit=1)[0]
        unique_types.add(its)
    context["unique_types"] = list(sorted(unique_types))

    # functions
    functions = []
    if hasattr(model_onnx, 'functions'):
        for fct in model_onnx.functions:
            opsets_fct = {}
            for oimp in fct.opset_import:
                opsets_fct[oimp.domain] = oimp.version
            functions.append(
                (fct.domain, fct.name,
                 {'proto': fct,
                  'opsets': opsets_fct}))
    context['functions'] = functions

    # node
    context['graph'] = graph

    # graph
    context['name'] = name or graph.name
    context['function_name'] = function_name
    if hasattr(model_onnx, 'graph'):
        context['doc_string'] = model_onnx.doc_string
    else:
        context['doc_string'] = ""

    # First rendering to detect any unused or replaced initializer.
    from jinja2 import Template  # delayed import
    template = Template(template)
    final = template.render(
        enumerate=enumerate, sorted=sorted, len=len, repr=repr,
        map=map, **context)

    final += "\n"
    if "\nreturn" in final:
        raise SyntaxError(
            "The produced code is wrong.\n%s" % final)
    if clean_code:
        cleaned_code = autopep8.fix_code(final, options=autopep_options)
        if "\nreturn" in cleaned_code:
            raise SyntaxError(
                "The cleaned code is wrong.\n%s\n------%s" % (
                    final, cleaned_code))
        return cleaned_code
    return final


def export2python(model_onnx, opset=None, verbose=True, name=None, rename=False,
                  autopep_options=None, function_name='main'):
    """
    Exports an ONNX model to the *python* syntax.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param autopep_options: :epkg:`autopep8` options
    :param function_name: main function name
    :return: python code

    The following example shows what a python code creating a graph
    implementing the KMeans would look like.

    .. runpython::
        :showcode:
        :process:

        import numpy
        from sklearn.cluster import KMeans
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.onnx_tools.onnx_export import export2python

        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X, target_opset=14)
        code = export2python(onx)

        print(code)
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    if not isinstance(model_onnx, ModelProto):
        raise TypeError(
            "The function expects a ModelProto not %r." % type(model_onnx))
    code = export_template(model_onnx, template=_template_python,
                           name=name, autopep_options=autopep_options,
                           clean_code=True, function_name=function_name)
    return code
