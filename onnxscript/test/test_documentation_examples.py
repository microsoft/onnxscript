"""
@brief      test log(time=60s)
"""
import unittest
import os
import sys
import subprocess


class TestDocumentationExample(unittest.TestCase):

    def test_documentation_examples(self):

        this = os.path.abspath(os.path.dirname(__file__))
        onxc = os.path.normpath(os.path.join(this, '..', '..'))
        pypath = os.environ.get('PYTHONPATH', None)
        sep = ";" if sys.platform == 'win32' else ':'
        pypath = "" if pypath in (None, "") else (pypath + sep)
        pypath += onxc
        os.environ['PYTHONPATH'] = pypath
        fold = os.path.normpath(
            os.path.join(this, '..', '..', 'docs', 'examples'))
        sys.path.insert(0, fold)
        found = os.listdir(fold)
        tested = 0
        for name in sorted(found):
            if os.path.splitext(name)[-1] != '.py':
                continue
            with self.subTest(name=name):
                cmds = [sys.executable, "-u", os.path.join(fold, name)]
                p = subprocess.Popen(
                    cmds, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                res = p.communicate()
                _, err = res
                st = err.decode('ascii', errors='ignore')
                if len(st) > 0 and 'Traceback' in st:
                    raise RuntimeError(  # pylint: disable=W0707
                        "Example '{}' (cmd: {} - exec_prefix="
                        "'{}') failed due to\n{}"
                        "".format(name, cmds, sys.exec_prefix, st))
                tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
