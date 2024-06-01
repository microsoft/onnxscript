# Adapted from
# https://github.com/pytorch/pytorch/blob/b505e8647547f029d0f7df408ee5f2968f757f89/test/test_public_bindings.py#L523
# Original code PyTorch license https://github.com/pytorch/pytorch/blob/main/LICENSE
# Modifications Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import importlib
import itertools
import os
import pathlib
import pkgutil
import unittest
from typing import Iterable

import onnxscript.ir

IR_NAMESPACE = "onnxscript.ir"


def _find_all_importables(pkg):
    """Find all importables in the project.
    Return them in order.
    """
    return sorted(
        set(
            itertools.chain.from_iterable(
                _discover_path_importables(pathlib.Path(p), pkg.__name__) for p in pkg.__path__
            ),
        ),
    )


def _discover_path_importables(pkg_path: os.PathLike, pkg_name: str) -> Iterable[str]:
    """Yield all importables under a given path and package.
    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages. Taken from https://stackoverflow.com/questions/41203765/init-py-required-for-pkgutil-walk-packages-in-python3
    """
    for dir_path, _, file_names in os.walk(pkg_path):
        pkg_dir_path = pathlib.Path(dir_path)

        if pkg_dir_path.parts[-1] == "__pycache__":
            continue

        if all(pathlib.Path(_).suffix != ".py" for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_path)
        pkg_pref = ".".join((pkg_name, *rel_pt.parts))
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
                (str(pkg_dir_path),),
                prefix=f"{pkg_pref}.",
            )
        )


def _is_mod_public(modname: str) -> bool:
    split_strs = modname.split(".")
    return all(not (elem.startswith("_") or "_test" in elem) for elem in split_strs)


def _validate_module(modname: str, failure_list: list[str]) -> None:
    mod = importlib.import_module(modname)
    if not _is_mod_public(modname):
        return

    # verifies that each public API has the correct module name and naming semantics
    def check_one_element(elem, modname, mod, *, is_public, is_all):
        obj = getattr(mod, elem)
        elem_module = getattr(obj, "__module__", None)
        # Only used for nice error message below
        why_not_looks_public = ""
        if elem_module is None:
            why_not_looks_public = "because it does not have a `__module__` attribute"
        elem_modname_starts_with_mod = (
            elem_module is not None
            and elem_module.startswith(IR_NAMESPACE)
            and "._" not in elem_module
        )
        if not why_not_looks_public and not elem_modname_starts_with_mod:
            why_not_looks_public = (
                f"because its `__module__` attribute (`{elem_module}`) is not within the "
                f"onnxscript.ir library or does not start with the submodule where it is defined (`{modname}`)"
            )
        # elem's name must NOT begin with an `_` and it's module name
        # SHOULD start with it's current module since it's a public API
        looks_public = not elem.startswith("_") and elem_modname_starts_with_mod
        if not why_not_looks_public and not looks_public:
            why_not_looks_public = f"because it starts with `_` (`{elem}`)"

        if is_public != looks_public:
            if is_public:
                why_is_public = (
                    f"it is inside the module's (`{modname}`) `__all__`"
                    if is_all
                    else "it is an attribute that does not start with `_` on a module that "
                    "does not have `__all__` defined"
                )
                fix_is_public = (
                    f"remove it from the modules's (`{modname}`) `__all__`"
                    if is_all
                    else f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                )
            else:
                assert is_all
                why_is_public = f"it is not inside the module's (`{modname}`) `__all__`"
                fix_is_public = f"add it from the modules's (`{modname}`) `__all__`"

            if looks_public:
                why_looks_public = (
                    "it does look public because it follows the rules from the doc above "
                    "(does not start with `_` and has a proper `__module__`)."
                )
                fix_looks_public = "make its name start with `_`"
            else:
                why_looks_public = why_not_looks_public
                if not elem_modname_starts_with_mod:
                    fix_looks_public = (
                        "make sure the `__module__` is properly set and points to a submodule "
                        f"of `{modname}`"
                    )
                else:
                    fix_looks_public = "remove the `_` at the beginning of the name"

            failure_list.append(f"# {modname}.{elem}:")
            is_public_str = "" if is_public else " NOT"
            failure_list.append(f"  - Is{is_public_str} public: {why_is_public}")
            looks_public_str = "" if looks_public else " NOT"
            failure_list.append(f"  - Does{looks_public_str} look public: {why_looks_public}")
            # Swap the str below to avoid having to create the NOT again
            failure_list.append(
                "  - You can do either of these two things to fix this problem:"
            )
            failure_list.append(f"    - To make it{looks_public_str} public: {fix_is_public}")
            failure_list.append(
                f"    - To make it{is_public_str} look public: {fix_looks_public}"
            )

    if hasattr(mod, "__all__"):
        public_api = mod.__all__
        all_api = dir(mod)
        for elem in all_api:
            check_one_element(elem, modname, mod, is_public=elem in public_api, is_all=True)
    else:
        all_api = dir(mod)
        for elem in all_api:
            if not elem.startswith("_"):
                check_one_element(elem, modname, mod, is_public=True, is_all=False)


class TestPublicApiNamespace(unittest.TestCase):
    tested_modules = (IR_NAMESPACE, *(_find_all_importables(onnxscript.ir)))

    def test_correct_module_names(self):
        """
        An API is considered public, if  its  `__module__` starts with `onnxscript.ir`
        and there is no name in `__module__` or the object itself that starts with "_".
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        """
        failure_list = []

        for modname in self.tested_modules:
            _validate_module(modname, failure_list)

        msg = (
            "Make sure that everything that is public is expected (in particular that the module "
            "has a properly populated `__all__` attribute) and that everything that is supposed to be public "
            "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        )

        msg += "\n\nFull list:\n"
        msg += "\n".join(failure_list)

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == "__main__":
    unittest.main()
