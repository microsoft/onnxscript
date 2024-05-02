# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
import sys
import unittest


class TestDocumentationExample(unittest.TestCase):
    def do_test_folder(self, folder):
        sys.path.insert(0, folder)
        found = os.listdir(folder)
        tested = 0
        for name in sorted(found):
            if os.path.splitext(name)[-1] != ".py":
                continue
            if __name__ == "__main__":
                print(f"run {name!r}")
            with self.subTest(folder=folder, name=name):
                cmds = [sys.executable, "-u", os.path.join(folder, name)]
                with subprocess.Popen(
                    cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as p:
                    res = p.communicate()
                    _, err = res
                    st = err.decode("ascii", errors="ignore")
                    if len(st) > 0 and "Traceback" in st:
                        raise RuntimeError(  # pylint: disable=W0707
                            f"Example '{name}' (cmd: {cmds} - exec_prefix='{sys.exec_prefix}') "
                            f"failed due to\n{st}"
                        )
                    tested += 1
        if tested == 0:
            raise RuntimeError(f"No example was tested in folder {folder}.")

    def test_documentation_examples(self):
        this = os.path.abspath(os.path.dirname(__file__))
        onxc = os.path.normpath(os.path.join(this, "..", ".."))
        pypath = os.environ.get("PYTHONPATH", None)
        sep = ";" if sys.platform == "win32" else ":"
        pypath = "" if pypath in (None, "") else (pypath + sep)
        pypath += onxc
        os.environ["PYTHONPATH"] = pypath

        def test(*relpath):
            self.do_test_folder(os.path.normpath(os.path.join(this, *relpath)))

        test("..", "..", "docs", "examples")
        test("..", "..", "docs", "tutorial", "examples")
        test("..", "..", "docs", "rewriter", "examples")


if __name__ == "__main__":
    unittest.main(verbosity=2)
