from __future__ import annotations

import functools
import logging

import onnx
from packaging import version

import onnxscript
import onnxscript._legacy_ir as ir
from onnxscript._legacy_ir import visitor
from onnxscript.rewriter import pattern

logger = logging.getLogger(__name__)


class FunctionRewriteError(RuntimeError): ...


@functools.lru_cache
def parse_domain(function_domain: str) -> tuple[str, version.Version | None]:
    splits = function_domain.split(".")
    if splits[0] != "pkg":
        raise FunctionRewriteError(
            f"Invalid domain: {function_domain}. Must start with 'pkg'."
        )
    splits = splits[1:]
    for i, s in enumerate(splits):
        if s.isdigit():
            return ".".join(splits[:i]), version.parse(".".join(splits[i:]))
    return ".".join(splits), None


MIN_VERSION = version.parse("0")
MAX_VERSION = version.parse("9999")


class VersionController:
    def __init__(self):
        # A dispatch table for rewrite implementation based on the function package version.
        self.dispatch_table: dict[tuple[version.Version, version.Version], callable] = {}

    def register_version(
        self,
        min_version: version.Version | str | None = None,
        max_version: version.Version | str | None = None,
    ):
        """Register a function implementation for a specific package version range [min_version, max_version).

        Args:
            min_version: The minimum version of the package. Inclusive.
            max_version: The maximum version of the package. Exclusive.
        """
        # TODO: check for version overloap

        min_version = MIN_VERSION if min_version is None else min_version
        max_version = MAX_VERSION if max_version is None else max_version
        if isinstance(min_version, str):
            min_version = version.parse(min_version)
        if isinstance(max_version, str):
            max_version = version.parse(max_version)

        def deco(func):
            self.dispatch_table[(min_version, max_version)] = func
            return func

        return deco

    def dispatch(self, version: version.Version | None) -> callable | None:
        if version is None:
            if len(self.dispatch_table) == 1:
                return next(iter(self.dispatch_table.values()))
            raise ValueError(
                "No function package version specified, however there are multiple "
                f"fusion rules based on package version: {self.dispatch_table.keys()}."
            )
        for (min_version, max_version), func in self.dispatch_table.items():
            greater_than_min = min_version is None or min_version <= version
            less_than_max = max_version is None or version < max_version
            if greater_than_min and less_than_max:
                return func
        return None


class FunctionRewriteRule(pattern.RewriteRule):
    FUNCTION_KEYWORD: str | tuple[str]
    """The keyword to match the function name. If a tuple, any keyword will match."""

    PACKAGE_NAME: str
    """The package name to match.

    For example, 'transformers' to match for domain name 'pkg.transformers.4.36.2'.
    """

    _opset_imports: dict[str, int]
    onnx_opset: onnxscript.values.Opset
    _function_shape_env: visitor.FunctionShapeEnv

    def __init__(self, opset: onnxscript.values.Opset = onnxscript.opset18) -> None:
        self.onnx_opset = opset

    def _match_function(self, function: onnx.FunctionProto, pkg_name: str) -> bool:
        # TODO: Consolidate more checks from `compose_new_function` to here.
        if pkg_name != self.PACKAGE_NAME:
            logger.info(
                "Rule %s did not match function %s::%s. Package name mismatch '%s' != '%s'.",
                self.__class__.__name__,
                function.domain,
                function.name,
                self.PACKAGE_NAME,
                pkg_name,
            )
            return False

        if isinstance(self.FUNCTION_KEYWORD, str):
            return function.name.find(self.FUNCTION_KEYWORD) != -1
        elif isinstance(self.FUNCTION_KEYWORD, tuple):
            return any(function.name.find(keyword) != -1 for keyword in self.FUNCTION_KEYWORD)
        else:
            raise ValueError(  # noqa: TRY004
                f"Function keyword must be str or tuple, got {self.FUNCTION_KEYWORD}"
            )

    def _find_node_contains_key_in_name(
        self, function: onnx.FunctionProto, keyword: str
    ) -> onnx.NodeProto | None:
        for node in function.node:
            if node.name.find(keyword) != -1:
                return node
        return None

    def _find_node_by_type(
        self, function: onnx.FunctionProto, domain: str, op_type: str
    ) -> onnx.NodeProto | None:
        # Repeat
        for node in function.node:
            if node.domain == domain and node.op_type == op_type:
                return node
        return None

    def _find_constant_node(
        self, function: onnx.FunctionProto, value_name: str
    ) -> onnx.NodeProto | None:
        # Potentially repeat, utility function.
        for node in function.node:
            for output in node.output:
                if output == value_name:
                    return node
        return None

    def compose_new_function(
        self, old_function: onnx.FunctionProto, pkg_version: version.Version | None
    ) -> tuple[onnx.FunctionProto, tuple[onnx.OperatorSetIdProto]]:
        """Compose a new function from the old function.

        Returns:
            A tuple of the new function and the opset imports.

        Raises:
            FunctionRewriteError: If the rewrite fails.
        """
        func = self._version_controller.dispatch(pkg_version)
        if func is not None:
            return func(self, old_function)
        raise FunctionRewriteError(
            f"No rewrite implementation for package version {pkg_version}."
        )

    def try_rewrite_function(
        self, function: onnx.FunctionProto, model: onnx.ModelProto
    ) -> bool:
        try:
            pkg_name, pkg_version = parse_domain(function.domain)
        except FunctionRewriteError as e:
            logger.warning("Could not parse domain: %s", e)
            return False

        if pkg_version is None and not pkg_name.startswith("onnxscript"):
            logger.warning(
                "Could not parse version for domain of function %s::%s. "
                "Usually this implies the model source is not from a package, but from arbitrary python files instead. "
                "For example, models not defined in huggingface/transformers but loaded via 'trust_remote_code=True'.",
                function.domain,
                function.name,
            )

        if not self._match_function(function, pkg_name):
            return False
        logger.info(
            "Rule %s matched function %s::%s",
            self.__class__.__name__,
            function.domain,
            function.name,
        )

        try:
            new_function, opset_imports = self.compose_new_function(function, pkg_version)
        except FunctionRewriteError as e:
            logger.warning("Could not rewrite function: %s", e)
            return False

        nodes = new_function.node

        del function.input[:]
        function.input.extend(new_function.input)
        del function.output[:]
        function.output.extend(new_function.output)

        del function.node[:]
        function.node.extend(nodes)
        for new_opset in opset_imports:
            function.opset_import.append(new_opset)
            if new_opset.domain not in self._opset_imports:
                model.opset_import.append(new_opset)

        return True

    def try_rewrite(self, model: ir.Model, value) -> bool:
        raise NotImplementedError(
            "Use `try_rewrite_function` instead for function based rewrites."
        )

    def lookup(self, function: onnx.FunctionProto, value_name: str) -> ir.Value | None:
        return self._function_shape_env.lookup(function, value_name)

    def apply_to_model(self, model: ir.Model, *, commute: bool = False) -> int:
        del commute  # unused
        model_proto: onnx.ModelProto = model.original_model_proto
        self._function_shape_env = visitor.FunctionShapeEnv()
        self._function_shape_env.load_from_model_proto(model.original_model_proto)
        self._opset_imports = {x.domain: x.version for x in model_proto.opset_import}

        rewrite_count = 0
        for function in model_proto.functions:
            rewrite_count += self.try_rewrite_function(function, model_proto)
        return rewrite_count

    def count_matches(self, model, *, commute: bool = False) -> int:
        raise NotImplementedError()

    def commute(self) -> list[pattern.RewriteRule]:
        raise NotImplementedError()
