# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy
import onnx
from onnx import FunctionProto, GraphProto, ModelProto, TensorProto, ValueInfoProto
from onnx.helper import make_node

import onnxscript.onnx_types
import onnxscript.type_annotation

kwlist = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


def _get_const_repr(const_node):
    """Given an ONNX Constant-op node, returns a string representation of
    the constant-value in ONNXScript, if a compact representation is possible.
    Returns None otherwise.
    Supports only FLOAT/INT64 values and scalars and small rank-1 tensors.
    This needs to be reconciled with the ONNXScript converter.
    """
    assert const_node.op_type == "Constant", "Expected a constant node"
    attr = const_node.attribute[0]
    if not attr.HasField("t"):
        return None
    tensor_proto = attr.t
    if tensor_proto.data_type in {TensorProto.FLOAT, TensorProto.INT64}:
        rank = len(tensor_proto.dims)
        if rank == 0:
            array = onnx.numpy_helper.to_array(tensor_proto).reshape(1)
            return repr(array[0])
        if rank == 1 and tensor_proto.dims[0] < 5:
            return repr(list(onnx.numpy_helper.to_array(tensor_proto)))
    return None


def _cleanup_variable_name(name: ValueInfoProto | str) -> Optional[str]:
    """Converts given name into a valid python variable names.
    Handles names that clash with python keywords and common issues seen in ONNX models:
    * Identifiers like "5" (that do not start with an alpha character)
    * Identifiers that contain a dot like "layers.0.foo"
    This is a simple heuristic, and doesn't guarantee it avoids name-clashes.
    Empty names, which are special in ONNX, are not renamed. A None is returned.
    """
    if isinstance(name, ValueInfoProto):
        # Handle graph/function input/output uniformly
        name = name.name
    assert isinstance(name, str)
    if name == "":
        return None
    if name in kwlist:
        return f"r_{name}"
    first = name[0]
    if not (first.isalpha() or (first == "_")):
        name = f"__{name}"

    def rename_char(char):
        """Replace invalid character by double underscore."""
        return char if (char.isalnum() or (char == "_")) else "__"

    return "".join([rename_char(c) for c in name])


def _make_short_name_mapper():
    """Returns a renamer used to create short new names  (like v0, v1, ...) for variables."""
    variable_names: dict[str, str] = {}

    def renamer(name):
        # TODO: simplify this. No need to use _cleanup_variable_name?
        var_name = _cleanup_variable_name(name)
        if var_name in variable_names:
            return variable_names[var_name]
        new_name = f"v{len(variable_names) + 1}"
        assert var_name is not None  # TODO(rama): This looks suspect.
        variable_names[var_name] = new_name
        return new_name

    return renamer


def _translate_type(onnx_type):
    """Converts a onnx type into a type defined by *onnxscript*."""
    return onnxscript.onnx_types.onnx_type_to_onnxscript_repr(onnx_type)


def _translate_signature(inputs, outputs):
    """Produce the script-functions signature."""

    def input_sig(inp: ValueInfoProto | str):
        if isinstance(inp, ValueInfoProto):
            # GraphProto inputs/outputs are ValueInfoProto
            return f"{_cleanup_variable_name(inp.name)}: {_translate_type(inp.type)}"

        # FunctionProto inputs/outputs are just strings
        return _cleanup_variable_name(inp)

    result = f"({', '.join([input_sig(x) for x in inputs])})"
    if outputs and isinstance(outputs[0], ValueInfoProto):
        result += f" -> ({', '.join([_translate_type(x.type) for x in outputs])})"
    return f"{result}:"


def _to_str(s):
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return s


def _is_attribute_ref(attr: onnx.AttributeProto) -> bool:
    return attr.HasField("ref_attr_name") and attr.ref_attr_name != ""


def _attribute_value(attr: onnx.AttributeProto):
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    if attr.type == onnx.AttributeProto.STRING:
        return _to_str(attr.s)
    if attr.type == onnx.AttributeProto.TENSOR:
        tensor_proto = attr.t
        if onnx.external_data_helper.uses_external_data(tensor_proto):
            return tensor_proto
        else:
            return onnx.numpy_helper.to_array(tensor_proto)
    # TODO:
    # - onnx.AttributeProto.GRAPH
    # - onnx.AttributeProto.SPARSE_TENSOR
    # - onnx.AttributeProto.TYPE_PROTO
    if attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == onnx.AttributeProto.STRINGS:
        return list(map(_to_str, attr.strings))
    # TODO:
    # - onnx.AttributeProto.TENSORS
    # - onnx.AttributeProto.GRAPHS
    # - onnx.AttributeProto.SPARSE_TENSORS
    # - onnx.AttributeProto.TYPE_PROTOS
    raise NotImplementedError(f"Unable to return a value for attribute {attr!r}.")


def _update_names_used_in_graph(names: set[str], graph: GraphProto) -> None:
    """Adds the names used in a graph to given set."""
    names.update(x.name for x in graph.input)
    names.update(x.name for x in graph.output)
    names.update(x.name for x in graph.initializer)
    for node in graph.node:
        _update_names_used_in_node(names, node)


def _update_names_used_in_node(names: set[str], node: onnx.NodeProto) -> None:
    names.update(node.input)
    names.update(node.output)
    for attr in node.attribute:
        if attr.HasField("g"):
            _update_names_used_in_graph(names, attr.g)
        for g in attr.graphs:
            _update_names_used_in_graph(names, g)


def _update_names_used_in_function(names: set[str], fun: FunctionProto) -> None:
    names.update(fun.input)
    names.update(fun.output)
    for node in fun.node:
        _update_names_used_in_node(names, node)


def _names_used_in_function(fun: FunctionProto) -> set[str]:
    names: set[str] = set()
    _update_names_used_in_function(names, fun)
    return names


class Exporter:
    """Class used for recursive traversal of Proto structures."""

    def __init__(
        self, rename: bool, use_operators: bool = False, inline_const: bool = False
    ) -> None:
        self.use_operators = use_operators
        if rename:
            rename_function = _make_short_name_mapper()
        else:
            rename_function = _cleanup_variable_name
        self._rename_variable = self._handle_attrname_conflict(rename_function)
        self.inline_const = inline_const
        self.constants: dict[str, str] = {}
        self._attr_renaming: dict[str, str | None] = {}  # For current function.
        self._names_used: set[str] = set()  # For current function.

    def _handle_attrname_conflict(self, renamer):
        """Add ref-attr-name-conflict handling logic to renaming function."""

        def new_renamer(name):
            new_name = renamer(name)
            if new_name not in self._attr_renaming:
                return new_name
            # Name conflicts with attribute parameter name.
            alternate = self._attr_renaming[new_name]
            if alternate is not None:
                return alternate
            counter = 0
            candidate = new_name
            while candidate in self._names_used:
                candidate = f"{new_name}_{counter}"
                counter += 1
            self._attr_renaming[new_name] = candidate
            self._names_used.add(candidate)
            return candidate

        return new_renamer

    def _rename_variable_s(self, name):
        """Renames all names equal to a python keyword."""
        return str(self._rename_variable(name))

    def _rename_domain(self, domain: str) -> str:
        if domain == "":
            return "opset"
        return domain.replace(".", "_")

    def _make_opset_name(self, domain, version):
        return f"{self._rename_domain(domain)}{version}"

    def _python_make_node_name(self, domain, version, name, node=False):
        name = _cleanup_variable_name(
            name
        )  # TODO: Is this a typo? Is it supposed to be self._rename_variable(name)?
        if node:
            if version is None:
                version = 1
            if not isinstance(version, int):
                raise TypeError(
                    f"version must be an integer not {version!r} for domain={domain!r} "
                    f"and name={name!r}."
                )
            opset = self._make_opset_name(domain, version)
            return f"{opset}.{name}"
        return name

    def _python_make_node_graph(self, graph, opsets, indent=0, output_names=None):
        """Translates a GraphProto into python."""
        code = []
        sindent = "    " * indent
        if hasattr(graph, "initializer"):
            for init in graph.initializer:
                node = make_node(
                    "Constant",
                    [],
                    [self._rename_variable(init.name)],  # type: ignore[list-item]
                    value=init,
                )
                code.append(self._python_make_node(node, opsets, indent=indent))
        if hasattr(graph, "sparse_initializer") and len(graph.sparse_initializer) > 0:
            raise NotImplementedError("Unable to convert sparse_initilizer into python.")
        for node in graph.node:
            pynode = self._python_make_node(node, opsets, indent=indent)
            if pynode:
                code.append(pynode)
        if output_names is not None:
            for fr, to in zip(graph.output, output_names):
                code.append(
                    f"{sindent}{self._rename_variable(to)} = {self._rename_variable(fr.name)}"
                )
        final = "\n".join(code)
        return final

    def _python_make_node_make_attribute_str(self, node):
        attributes = []
        for at in node.attribute:
            if _is_attribute_ref(at):
                attributes.append((at.name, at.ref_attr_name))
                continue
            value = _attribute_value(at)
            if isinstance(value, str):
                attributes.append((at.name, f"{value!r}"))
                continue
            if isinstance(value, numpy.ndarray):
                onnx_dtype = at.t.data_type
                if len(value.shape) == 0:
                    text = (
                        f'make_tensor("value", {onnx_dtype}, dims=[], '
                        f"vals=[{value.tolist()!r}])"
                    )
                else:
                    text = (
                        f'make_tensor("value", {onnx_dtype}, dims={list(value.shape)!r}, '
                        f"vals={value.ravel().tolist()!r})"
                    )
                attributes.append((at.name, text))
                continue
            if isinstance(value, TensorProto):
                metadata = onnx.external_data_helper.ExternalDataInfo(value)
                name = value.name or "value"
                text = "external_tensor("
                text += f"{name!r}, {value.data_type}, {list(value.dims)!r}"
                text += f", {metadata.location!r}"
                if metadata.offset:
                    text += f", offset={metadata.offset!r}"
                if metadata.length:
                    text += f", length={metadata.length!r}"
                attributes.append((at.name, text))
                continue
            attributes.append((at.name, repr(value)))

        return ", ".join(f"{k}={v}" for k, v in attributes)

    def _python_make_node_if(self, node, opsets, indent=0):
        """Translates a node If into python."""
        sindent = "    " * indent
        code = [f"{sindent}if {node.input[0]}:"]
        if len(node.attribute) != 2:
            raise RuntimeError(
                f"Node {node.op_type!r} expected two attributes not {len(node.attribute)}."
            )
        atts = node.attribute
        if atts[0].name == "else_branch":
            else_branch, then_branch = atts[0].g, atts[1].g
        else:
            else_branch, then_branch = atts[1].g, atts[0].g
        code.append(
            self._python_make_node_graph(
                then_branch, opsets, indent=indent + 1, output_names=node.output
            )
        )
        code.append(f"{sindent}else:")
        code.append(
            self._python_make_node_graph(
                else_branch, opsets, indent=indent + 1, output_names=node.output
            )
        )
        return "\n".join(code)

    def _python_make_node_loop(self, node, opsets, indent=0):
        """Translates a node Loop into python."""
        body = node.attribute[0].g
        sindent = "    " * indent
        n_iter = self._rename_variable(node.input[0])
        cond = self._rename_variable(node.input[1])
        # v_initial = node.input[2]
        rows = []
        if n_iter and not cond:
            rows.append(f"{sindent}for {body.input[0].name} in range({n_iter}):")
        elif not n_iter and cond:
            rows.append(f"{sindent}while {cond}:")
        elif n_iter and cond:
            rows.append(f"{sindent}for {body.input[0].name} in range({n_iter}):")
            rows.append(f"{sindent}    if not {cond}:")
            rows.append(f"{sindent}        break")
        else:
            raise RuntimeError(
                f"Unable to export loop type {node.op_type!r} into python because "
                "there is no stop condition."
            )
        rows.append(
            self._python_make_node_graph(
                body, opsets, indent=indent + 1, output_names=node.output
            )
        )
        return "\n".join(rows)

    def _python_make_node_scan(self, node, opsets, indent=0):
        """Translates a node Scan into python."""
        raise NotImplementedError()

    def _lookup(self, var):
        if var in self.constants:
            return self.constants[var]

        return self._rename_variable_s(var)

    def _python_make_node(self, onnx_node, opsets, indent=0):
        if isinstance(onnx_node, dict):
            node = onnx_node["onnx_node"]
        else:
            node = onnx_node
        if self.inline_const and node.op_type == "Constant":
            val = _get_const_repr(node)
            if val is not None:
                self.constants[node.output[0]] = str(val)
                return ""
        if node.op_type in {"If", "Loop", "Scan"}:
            # If, Loop, Scan
            if node.op_type == "If":
                return self._python_make_node_if(node, opsets, indent=indent)
            if node.op_type == "Loop":
                return self._python_make_node_loop(node, opsets, indent=indent)
            if node.op_type == "Scan":
                return self._python_make_node_scan(node, opsets, indent=indent)
            raise RuntimeError(f"Unable to export node type {node.op_type!r} into python.")
        if any(hasattr(att, "g") and att.g and att.g.ByteSize() > 0 for att in node.attribute):
            raise RuntimeError(f"Unable to export node type {node.op_type!r} into python.")
        ops = {
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "MatMul": "@",
            "Div": "/",
            "Pow": "**",
            "And": "&",
            "Or": "|",
            "Greater": ">",
            "Equal": "==",
            "Lesser": "<",
            "GreaterOrEqual": ">=",
            "LessOrEqual": "<=",
        }
        sindent = "    " * indent
        if self.use_operators and node.op_type in ops:
            return (
                f"{sindent}{self._rename_variable(node.output[0])} = "
                f"{(f' {ops[node.op_type]} ').join(map(self._lookup, node.input))}"
            )
        name = self._python_make_node_name(
            node.domain, opsets[node.domain], node.op_type, node=True
        )
        attributes_str = self._python_make_node_make_attribute_str(node)
        if len(node.input) > 0 and len(attributes_str) > 0:
            attributes_str = f", {attributes_str}"
        output_names: list[Any] = []
        for i, o in enumerate(node.output):
            if o in ("", None):
                output_names.append(f"_{i}")
            else:
                output_names.append(self._rename_variable(o))

        text = [
            sindent,
            ", ".join(output_names),
            " = ",
            name,
            "(",
            ", ".join(map(self._lookup, node.input)),
            attributes_str,
            ")",
        ]
        return "".join(text)

    def _translate_opset_import(self, domain: str, version: int) -> str:
        if domain in {"", "ai.onnx"}:
            return f"from onnxscript.onnx_opset import opset{version}\n"
        else:
            varname = self._make_opset_name(domain, version)
            return f"{varname} = Opset('{domain}', {version})\n"

    def _translate_opset_imports(
        self, opset_imports: Sequence[onnx.OperatorSetIdProto]
    ) -> str:
        return "".join(
            [self._translate_opset_import(x.domain, x.version) for x in opset_imports]
        )

    def _translate_opset_imports_of(
        self, proto: ModelProto | FunctionProto | GraphProto
    ) -> str:
        if hasattr(proto, "opset_import"):
            text = self._translate_opset_imports(proto.opset_import)
            if isinstance(proto, FunctionProto):
                if not any(x.domain == proto.domain for x in proto.opset_import):
                    text += self._translate_opset_import(proto.domain, 1)
            return text
        return ""

    def _translate_function_signature(self, funproto: onnx.FunctionProto) -> str:
        """Generate signature for FunctionProto."""
        type_map = _attribute_param_types(funproto)

        def attr_sig(attr_name: str) -> str:
            self._attr_renaming[attr_name] = None
            self._names_used.add(attr_name)
            # A default type of INT is used for attribute parameters that are never used.
            type = type_map.get(attr_name, onnx.AttributeProto.INT)
            typerep = onnxscript.type_annotation.onnx_attr_type_to_onnxscript_repr(type)
            return f"{attr_name}: {typerep}"

        inputs = [self._rename_variable(x) for x in funproto.input]
        attrs = [attr_sig(x) for x in funproto.attribute]
        input_and_attrs = ", ".join(inputs + attrs)  # type: ignore[arg-type]
        if len(funproto.attribute_proto) > 0:
            message = "\n   # Attribute parameters default-values not handled yet."
        else:
            message = ""
        return f"({input_and_attrs}):{message}"

    def _translate_function(self, funproto: onnx.FunctionProto) -> str:
        """Generate python code for FunctionProto."""
        opsets = {}
        for imported in funproto.opset_import:
            opsets[imported.domain] = imported.version
        self._attr_renaming = {}
        used_proto_names = _names_used_in_function(funproto)
        renamed_names_used = [self._rename_variable(x) for x in used_proto_names]
        self._names_used = set(renamed_names_used)
        result = []

        def add_line(line: str) -> None:
            result.append(line)

        opset_name = self._make_opset_name(funproto.domain, 1)
        add_line(f"@script({opset_name})")
        fun_name = self._python_make_node_name(funproto.domain, 1, funproto.name)
        fun_sig = self._translate_function_signature(funproto)
        add_line(f"def {fun_name}{fun_sig}")
        if funproto.doc_string:
            add_line(f'    """{funproto.doc_string}"""')
        for node in funproto.node:
            add_line(self._python_make_node(node, opsets, indent=1))
        return_values = ", ".join(self._rename_variable(x) for x in funproto.output)
        add_line(f"    return {return_values}")
        return "\n".join(result)

    def _translate_graph(self, model: onnx.ModelProto, function_name: str) -> str:
        graph = model.graph
        opsets = {}
        for imported in model.opset_import:
            opsets[imported.domain] = imported.version

        result: list[str] = []

        def add(line: str) -> None:
            result.append(line)

        add("@script()")
        add(f"def {function_name}{_translate_signature(graph.input, graph.output)}")
        doc = graph.doc_string
        if doc:
            add(f'    """{doc}"""')
        add(self._python_make_node_graph(graph, opsets, indent=1))
        return_values = ", ".join(self._rename_variable(x) for x in graph.output)
        add(f"    return {return_values}")
        return "\n".join(result)

    def _import_onnx_types(
        self, proto: onnx.ModelProto | onnx.GraphProto | onnx.FunctionProto
    ) -> str:
        """Generate import statements for types used in the graph."""
        if isinstance(proto, ModelProto):
            graph_or_function = proto.graph
        else:
            graph_or_function = proto
        used_types: set[str] = set()
        for t in list(graph_or_function.input) + list(graph_or_function.output):
            if hasattr(t, "type"):
                ts = _translate_type(t.type)
                its = ts.split("[", maxsplit=1)[0]
                used_types.add(its)
        # TODO: handle types in nested graphs.
        sorted_types = sorted(used_types)
        if sorted_types:
            return "from onnxscript.onnx_types import " + ", ".join(sorted_types)
        return ""

    def export(self, proto: onnx.ModelProto | onnx.FunctionProto, function_name: str) -> str:
        result: list[str] = []

        def add(line: str) -> None:
            result.append(line)

        # Generic imports.
        add("import numpy")
        add("from onnx import TensorProto")
        add("from onnx.helper import make_tensor")
        add("from onnxscript import script, external_tensor")
        add("from onnxscript.values import Opset")
        add(self._import_onnx_types(proto))

        if isinstance(proto, ModelProto):
            translated_functions = [self._translate_function(f) for f in proto.functions]
            translated_functions.append(self._translate_graph(proto, function_name))
        else:
            assert isinstance(proto, FunctionProto)
            # TODO: use function_name?
            translated_functions = [self._translate_function(proto)]

        # TODO: unique_function_domain_version.add((f.domain, 1))
        add(self._translate_opset_imports_of(proto))
        result.extend(translated_functions)

        add("")
        final = "\n".join(result)

        if "\nreturn" in final:
            raise SyntaxError(f"The produced code is wrong.\n{final}")

        return final


def _attribute_param_types(
    funproto: onnx.FunctionProto,
) -> dict[str, onnx.AttributeProto.AttributeType]:
    """Compute mapping from (names of) attribute parameters of function to their types."""
    type_map = {}

    def visit_node(node: onnx.NodeProto) -> None:
        for attr in node.attribute:
            if _is_attribute_ref(attr):
                type_map[attr.ref_attr_name] = attr.type
            elif attr.type == onnx.AttributeProto.GRAPH:
                visit_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for graph in attr.graphs:
                    visit_graph(graph)

    def visit_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            visit_node(node)

    for node in funproto.node:
        visit_node(node)
    return type_map


def export2python(
    model_onnx,
    opset=None,
    verbose=True,
    name=None,
    rename=False,
    function_name="main",
    use_operators=False,
    inline_const: bool = False,
):
    """Exports an ONNX model to the *python* syntax.

    Args:
        model_onnx: string or ONNX graph
        opset: opset to export to (None to select the one from the
            graph)
        verbose: inserts prints
        name: to overwrite onnx name
        rename: rename the names to get shorter names
        function_name: main function name
        use_operators: use Python operators.
        inline_const: replace ONNX constants inline if compact

    Returns:
        python code
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
    del opset  # unused
    del verbose  # unused
    del name  # unused
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    if not isinstance(model_onnx, (ModelProto, FunctionProto)):
        raise TypeError(f"The function expects a ModelProto not {type(model_onnx)!r}.")

    exporter = Exporter(rename, use_operators, inline_const)
    return exporter.export(model_onnx, function_name)
