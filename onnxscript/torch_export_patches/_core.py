import contextlib
import functools
import importlib
import re
from collections.abc import Generator
from typing import Any, Callable

import torch

from onnxscript.torch_export_patches import (
    _patch_torch,
    _patch_transformers,
    _treenode_registry,
)


def patched_function(name: str) -> tuple[type, Callable]:
    """Returns the module and the function needed to be patched."""
    spl = name.split(".")
    module_name = ".".join(spl[:-1])
    fname = spl[-1]
    module = importlib.import_module(module_name)
    if not hasattr(module, fname):
        return None, None
    return module, getattr(module, fname)


@functools.lru_cache
def get_patches(module: torch.nn.Module) -> tuple[str, list[Any]]:
    """Returns the list of patches to make for a specific nn.module."""
    to_patch = []
    for k in dir(module):
        if k.startswith("patched_"):
            v = getattr(module, k)
            if hasattr(v, "_PATCHED_CLASS_") and hasattr(v, "_PATCHES_"):
                to_patch.append(v)
            else:
                # a function
                doc = v.__doc__.lstrip()
                if doc.startswith("manual patch"):
                    continue
                reg = re.compile("[\\[]patch:([a-z_A-Z.]+)[\\]]")
                fall = reg.findall(doc)
                assert len(fall) == 1, (
                    f"Unable to find patching information for {v} in \n{doc}"
                )
                fmod, f = patched_function(fall[0])
                if fmod is None and f is None:
                    # The function does not exist in this version of transformers.
                    # No patch is needed.
                    continue
                to_patch.append({"module": fmod, "function": f, "patch": v})

    name = module.__name__
    return name, to_patch


def patch_module_or_class(mod, verbose: int = 0) -> dict[type, dict[type, Callable]]:
    """Applies all patches defined in classes prefixed by ``patched_``.

    ``cls._PATCHED_CLASS_`` defines the class to patch,
    ``cls._PATCHES_`` defines the method to patch.
    The returns information needs to be sent to :func:`unpatch_module_or_class`
    to revert the changes.

    Args:
        mod: module of list of clsses to patch
        verbose: verbosity

    Returns:
        patch info
    """
    if isinstance(mod, list):
        to_patch = mod
        name = "list"
    else:
        name, to_patch = get_patches(mod)

    res = {}
    for cls in to_patch:
        if isinstance(cls, dict):
            # a function
            keep = {}
            original = cls["module"]
            f = cls["function"]
            res[f] = f
            if verbose:
                print(f"[patch_module_or_class] function: {original.__name__}.{f.__name__}")
            setattr(original, f.__name__, cls["patch"])
            continue

        original = cls._PATCHED_CLASS_
        methods = cls._PATCHES_
        if verbose:
            print(f"[patch_module_or_class] {name}.{cls.__name__}: {', '.join(methods)}")

        keep = {n: getattr(original, n, None) for n in methods}
        for n in methods:
            setattr(original, n, getattr(cls, n))
        res[cls] = keep

    return res


def unpatch_module_or_class(mod, info: dict[type, dict[type, Callable]], verbose: int = 0):
    """Reverts modification made by :func:`patch_module_or_class`.

    Args:
        mod: module of list of clsses to patch
        info: patch information returned by patch_module_or_class
        verbose: verbosity
    """
    if isinstance(mod, list):
        to_patch = mod
        name = "list"
    else:
        name, to_patch = get_patches(mod)

    set_patch_cls = {i for i in to_patch if not isinstance(i, dict)}
    dict_patch_fct = {i["function"]: i for i in to_patch if isinstance(i, dict)}

    for cls, methods in info.items():
        if cls in set_patch_cls:
            if verbose:
                print(f"[unpatch_module_or_class] {name}.{cls.__name__}: {', '.join(methods)}")
            original = cls._PATCHED_CLASS_
            for n, v in methods.items():
                if v is None:
                    # The method did not exist. We remove it.
                    delattr(original, n)
                else:
                    setattr(original, n, v)
            continue
        assert cls in dict_patch_fct, (
            f"No patch registered for {cls} in {mod} (found {set_patch_cls} and {set(dict_patch_fct)})"
        )
        patch = dict_patch_fct[cls]
        if verbose:
            print(
                f"[unpatch_module_or_class] function {patch['module'].__name__}.{cls.__name__}"
            )
        setattr(patch["module"], cls.__name__, patch["function"])


@contextlib.contextmanager
def torch_export_patches(
    patch_sympy: bool = True,
    patch_torch: bool = True,
    patch_transformers: bool = False,
    patch_diffusers: bool = False,
    verbose: int = 0,
    patch: bool = True,
    custom_patches: type["torch.nn.Module"] | None = None,
    rewrite: list[Callable] | None = None,
    dump_rewriting: str | None = None,
) -> Generator[Any, Any, Any]:
    """Bypass the situations :func:`torch.export.export` does not support.

    See also :ref:`l-patches-explained` and :ref:`l-patch-coverage`.

    Args:
        patch_sympy: fix missing method ``name`` for IntegerConstant
        patch_torch: patches :epkg:`torch` with supported implementation
        patch_transformers: patches :epkg:`transformers` with supported implementation
        patch_diffusers: patches :epkg:`diffusers` with supported implementation
        patch: if False, disable all patches but keeps the registration of
            serialization functions if other patch functions are enabled
        custom_patches: to apply custom patches,
            every patched class must define static attributes
            ``_PATCHES_``, ``_PATCHED_CLASS_``
        rewrite: list of methods to automatically rewrite
            before exporting, methods with control flow need to be rewritten
            before being exported if the execution path depends on the inputs,
            this is done by function :func:`transform_method
            <onnx_diagnostic.torch_export_patches.patch_module.transform_method>`,
            its documentation provides possible values
        dump_rewriting: dumps rewriting information in file beginning with that prefix
        verbose: to show which patches is applied

    The list of available patches:

    * ``torch._dynamo.mark_static_address``
    * ``torch._subclasses.fake_impls.infer_size``
    * ``torch.vmap``
    * fix missing method ``name`` for ``sympy.S.IntegerConstant``
    * ``AttentionMaskConverter._make_causal_mask``
    * Serialization of ``MambaCache`` (in :epkg:`transformers`)
    * Serialization of ``DynamicCache`` (in :epkg:`transformers`)
    * reduce errors due to shape inference
    * fixes some transformers classes

    Serialization issues happen when a module takes one input or output
    has a type :func:`torch.export.export` cannot serialize.

    Examples:
        .. code-block:: python

            with torch_export_patches(patch_transformers=True) as modificator:
                inputs = modificator(inputs)
                onx = to_onnx(..., inputs, ...)

        .. code-block:: python

            with torch_export_patches(patch_transformers=True) as modificator:
                inputs = modificator(inputs)
                onx = torch.onnx.export(..., inputs, ...)

        It can be used as well to fix the torch export:

        .. code-block:: python

            with torch_export_patches(patch_transformers=True) as modificator:
                inputs = modificator(inputs)
                ep = torch.export.export(..., inputs, ...)

        When exporting a model with a cache, the following error message
        may appear ``AssertionError: Mutating module attribute _seen_tokens during export.``.
        It can be avoided by setting ``strict=False`` when call :func:`torch.export.export`.
    """
    if rewrite is not None or dump_rewriting is not None:
        raise ValueError(
            "The argument `rewrite` is not supported at the moment, as the automatic rewriting is not implemented yet."
        )
        # from .patch_module import torch_export_rewrite

        # with torch_export_rewrite(
        #     rewrite=rewrite, dump_rewriting=dump_rewriting, verbose=verbose
        # ), torch_export_patches(  # type: ignore[var-annotated]
        #     patch_sympy=patch_sympy,
        #     patch_torch=patch_torch,
        #     patch_transformers=patch_transformers,
        #     patch_diffusers=patch_diffusers,
        #     catch_constraints=catch_constraints,
        #     stop_if_static=stop_if_static,
        #     verbose=verbose,
        #     patch=patch,
        #     custom_patches=custom_patches,
        # ) as f:
        #     try:
        #         yield f
        #     finally:
        #         pass
    # elif not patch:
    if not patch:
        # NOTE: registering caches serialization is not recognized as patches
        # It's needed to successfully export the models through torch.export,
        # as the serialization functions are not registered by default.

        fct_callable = lambda x: x  # noqa: E731
        done = _treenode_registry.register_cache_serialization(
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            verbose=verbose,
        )
        try:
            yield fct_callable
        finally:
            _treenode_registry.unregister_cache_serialization(done, verbose=verbose)
    else:
        import torch
        import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
        import torch.jit

        if verbose:
            print(
                "[torch_export_patches] replace torch.jit.isinstance, torch._dynamo.mark_static_address"
            )

        ########
        # caches
        ########

        cache_done = _treenode_registry.register_cache_serialization(
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            verbose=verbose,
        )

        #############
        # patch sympy
        #############

        # TODO (anyone): Is this still needed?
        if patch_sympy:
            import sympy

            f_sympy_name = getattr(sympy.core.numbers.IntegerConstant, "name", None)

            if verbose:
                print(f"[torch_export_patches] sympy.__version__={sympy.__version__!r}")
                print("[torch_export_patches] patch sympy")

            sympy.core.numbers.IntegerConstant.name = lambda self: f"IntCst{self!s}"

        ###############
        # patch pytorch
        ###############

        if patch_torch:
            if verbose:
                print(f"[torch_export_patches] torch.__version__={torch.__version__!r}")
                print("[torch_export_patches] patch pytorch")

            # torch.vmap
            f_vmap = torch.vmap
            torch.vmap = _patch_torch.patched_vmap

            # TODO(anyone): Test these to see if they are still needed
            # torch._dynamo.mark_static_address
            f_mark_static_address = torch._dynamo.mark_static_address
            torch._dynamo.mark_static_address = lambda *_, **y_: None
            # torch._subclasses.fake_impls.infer_size
            f_infer_size = torch._subclasses.fake_impls.infer_size
            torch._subclasses.fake_impls.infer_size = _patch_torch.patched_infer_size
            # torch._refs._broadcast_shapes
            f__broadcast_shapes = torch._refs._broadcast_shapes
            torch._refs._broadcast_shapes = _patch_torch.patched__broadcast_shapes
            torch._meta_registrations._broadcast_shapes = (
                _patch_torch.patched__broadcast_shapes
            )

        ####################
        # patch transformers
        ####################

        if patch_transformers:
            try:
                from transformers import masking_utils
            except ImportError:
                masking_utils = None

            if verbose:
                import transformers

                print(
                    f"[torch_export_patches] transformers.__version__={transformers.__version__!r}"
                )
            revert_patches_info = patch_module_or_class(_patch_transformers, verbose=verbose)

            if (
                masking_utils
                and _patch_transformers.patch_masking_utils
                and hasattr(masking_utils, "_vmap_for_bhqkv")
            ):
                if verbose:
                    print(
                        "[torch_export_patches] patches transformers.masking_utils._vmap_for_bhqkv"
                    )
                f_transformers__vmap_for_bhqkv = masking_utils._vmap_for_bhqkv
                masking_utils._vmap_for_bhqkv = _patch_transformers.patched__vmap_for_bhqkv

            if (
                masking_utils
                and _patch_transformers.patch_masking_utils
                and hasattr(masking_utils, "eager_mask")
            ):
                if verbose:
                    print(
                        "[torch_export_patches] patches transformers.masking_utils.eager_mask"
                    )
                f_transformers_eager_mask = masking_utils.eager_mask
                masking_utils.eager_mask = _patch_transformers.patched_eager_mask
                if (
                    "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
                    and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
                    == f_transformers_eager_mask
                ):
                    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = (
                        _patch_transformers.patched_eager_mask
                    )

        if custom_patches:
            if verbose:
                print("[torch_export_patches] applies custom patches")
            revert_custom_patches_info = patch_module_or_class(custom_patches, verbose=verbose)

        ########
        # export
        ########

        def fct_callable(x):
            return x

        if verbose:
            print("[torch_export_patches] done patching")

        try:
            yield fct_callable
        finally:
            #######
            # sympy
            #######

            if verbose:
                print("[torch_export_patches] remove patches")

            if patch_sympy:
                # tracked by https://github.com/pytorch/pytorch/issues/143494
                if f_sympy_name:
                    sympy.core.numbers.IntegerConstant.name = f_sympy_name
                else:
                    delattr(sympy.core.numbers.IntegerConstant, "name")

                if verbose:
                    print("[torch_export_patches] restored sympy functions")

            #######
            # torch
            #######

            if patch_torch:
                # this should disappear when torch.jit is removed
                torch.vmap = f_vmap
                torch._dynamo.mark_static_address = f_mark_static_address
                # tracked by https://github.com/pytorch/pytorch/issues/143495
                torch._subclasses.fake_impls.infer_size = f_infer_size
                torch._refs._broadcast_shapes = f__broadcast_shapes
                torch._meta_registrations._broadcast_shapes = f__broadcast_shapes

                if verbose:
                    print("[torch_export_patches] restored pytorch functions")

            if custom_patches:
                if verbose:
                    print("[torch_export_patches] unpatches custom patches")
                unpatch_module_or_class(
                    custom_patches, revert_custom_patches_info, verbose=verbose
                )

            ##############
            # transformers
            ##############

            if patch_transformers:
                try:
                    # masking_utils is available only if transformers==4.53.0 or later
                    from transformers import masking_utils
                except ImportError:
                    masking_utils = None
                if verbose:
                    print("[torch_export_patches] unpatches transformers")
                unpatch_module_or_class(
                    _patch_transformers, revert_patches_info, verbose=verbose
                )

                if (
                    masking_utils
                    and _patch_transformers.patch_masking_utils
                    and hasattr(masking_utils, "_vmap_for_bhqkv")
                ):
                    masking_utils._vmap_for_bhqkv = f_transformers__vmap_for_bhqkv
                    if verbose:
                        print(
                            "[torch_export_patches] restored transformers.masking_utils._vmap_for_bhqkv"
                        )

                if (
                    masking_utils
                    and _patch_transformers.patch_masking_utils
                    and hasattr(masking_utils, "eager_mask")
                ):
                    f_transformers_eager_mask = masking_utils.eager_mask
                    masking_utils.eager_mask = f_transformers_eager_mask
                    if (
                        "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
                        and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
                        == _patch_transformers.patched_eager_mask
                    ):
                        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = (
                            f_transformers_eager_mask
                        )
                    if verbose:
                        print(
                            "[torch_export_patches] restored transformers.masking_utils.eager_mask"
                        )

            ########
            # caches
            ########

            _treenode_registry.unregister_cache_serialization(cache_done, verbose=verbose)
