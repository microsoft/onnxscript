# SPDX-License-Identifier: Apache-2.0

"""
Tests examples from the documentation.
"""
import unittest
import os
import sys
import importlib
import subprocess


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(
                module_name, module_file_path))
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(unittest.TestCase):

    def test_documentation_examples(self):

        this = os.path.abspath(os.path.dirname(__file__))
        folds = [os.path.normpath(os.path.join(this, '..', 'tutorial', 'examples'))]
        tested = 0
        for fold in folds:
            found = os.listdir(fold)
            for name in found:
                if not name.endswith(".py"):
                    continue
                with self.subTest(name=name, fold=fold):
                    if __name__ == "__main__":
                        print("run %r" % name)
                    try:
                        mod = import_source(fold, os.path.splitext(name)[0])
                        assert mod is not None
                    except FileNotFoundError:
                        # try another way
                        cmds = [sys.executable, "-u",
                                os.path.join(fold, name)]
                        p = subprocess.Popen(
                            cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        res = p.communicate()
                        out, err = res
                        st = err.decode('ascii', errors='ignore')
                        if len(st) > 0 and 'Traceback' in st:
                            raise RuntimeError(
                                f"Example {name!r} (cmd: {cmds!r} - "
                                f"exec_prefix={sys.exec_prefix!r}) "
                                f"failed due to\n{st}")
                    tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
