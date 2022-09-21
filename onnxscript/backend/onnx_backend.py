# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import os
import textwrap

import numpy
import onnx
from numpy import object as dtype_object
from numpy.testing import assert_almost_equal
from onnx.backend.test import __file__ as backend_folder
from onnx.numpy_helper import to_array, to_list


def assert_almost_equal_string(expected, value):
    """
    Compares two arrays knowing they contain strings.
    Raises an exception if the test fails.

    :param expected: expected array
    :param value: value
    """

    def is_float(x):
        try:
            return True
        except ValueError:  # pragma: no cover
            return False

    if all(map(is_float, expected.ravel())):
        expected_float = expected.astype(numpy.float32)
        value_float = value.astype(numpy.float32)
        assert_almost_equal(expected_float, value_float)
    else:
        assert_almost_equal(expected, value)


class OnnxBackendTest:
    """
    Definition of a backend test. It starts with a folder,
    in this folder, one onnx file must be there, then a subfolder
    for each test to run with this model.

    :param folder: test folder
    :param onnx_path: onnx file
    :param onnx_model: loaded onnx file
    :param tests: list of test
    """

    @staticmethod
    def _sort(filenames):
        temp = []
        for f in filenames:
            name = os.path.splitext(f)[0]
            i = name.split("_")[-1]
            temp.append((int(i), f))
        temp.sort()
        return [_[1] for _ in temp]

    @staticmethod
    def _read_proto_from_file(full):
        if not os.path.exists(full):
            raise FileNotFoundError("File not found: %r." % full)  # pragma: no cover
        with open(full, "rb") as f:
            serialized = f.read()
        try:
            loaded = to_array(onnx.load_tensor_from_string(serialized))
        except Exception as e:  # pylint: disable=W0703
            seq = onnx.SequenceProto()
            try:
                seq.ParseFromString(serialized)
                loaded = to_list(seq)
            except Exception:  # pylint: disable=W0703
                try:
                    loaded = onnx.load_model_from_string(serialized)
                except Exception:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to read %r, error is %s, content is %r."
                        % (full, e, serialized[:100])
                    ) from e
        return loaded

    @staticmethod
    def _load(folder, names):
        res = []
        for name in names:
            full = os.path.join(folder, name)
            new_tensor = OnnxBackendTest._read_proto_from_file(full)
            if isinstance(new_tensor, (numpy.ndarray, onnx.ModelProto, list)):
                t = new_tensor
            elif isinstance(new_tensor, onnx.TensorProto):
                t = to_array(new_tensor)
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unexpected type %r for %r." % (type(new_tensor), full)
                )
            res.append(t)
        return res

    def __repr__(self):
        "usual"
        return "%s(%r)" % (self.__class__.__name__, self.folder)

    def __init__(self, folder):
        if not os.path.exists(folder):
            raise FileNotFoundError(  # pragma: no cover
                "Unable to find folder %r." % folder
            )
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {".onnx"}]
        if len(onx) != 1:
            raise ValueError(  # pragma: no cover
                "There is more than one onnx file in %r (%r)." % (folder, onx)
            )
        self.folder = folder
        self.onnx_path = os.path.join(folder, onx[0])
        self.onnx_model = onnx.load(self.onnx_path)

        self.tests = []
        for sub in content:
            full = os.path.join(folder, sub)
            if os.path.isdir(full):
                pb = [c for c in os.listdir(full) if os.path.splitext(c)[-1] in {".pb"}]
                inputs = OnnxBackendTest._sort(c for c in pb if c.startswith("input_"))
                outputs = OnnxBackendTest._sort(
                    c for c in pb if c.startswith("output_")
                )

                t = dict(
                    inputs=OnnxBackendTest._load(full, inputs),
                    outputs=OnnxBackendTest._load(full, outputs),
                )
                self.tests.append(t)

    @property
    def name(self):
        "Returns the test name."
        return os.path.split(self.folder)[-1]

    def __len__(self):
        "Returns the number of tests."
        return len(self.tests)

    def _compare_results(self, index, i, e, o, decimal=None):
        """
        Compares the expected output and the output produced
        by the runtime. Raises an exception if not equal.

        :param index: test index
        :param i: output index
        :param e: expected output
        :param o: output
        :param decimal: precision
        """
        if isinstance(e, numpy.ndarray):
            if isinstance(o, numpy.ndarray):
                if decimal is None:
                    if e.dtype == numpy.float32:
                        deci = 6
                    elif e.dtype == numpy.float64:
                        deci = 12
                    else:
                        deci = 7
                else:
                    deci = decimal
                if e.dtype == dtype_object:
                    try:
                        assert_almost_equal_string(e, o)
                    except AssertionError as ex:
                        raise AssertionError(  # pragma: no cover
                            "Output %d of test %d in folder %r failed."
                            % (i, index, self.folder)
                        ) from ex
                else:
                    try:
                        assert_almost_equal(e, o, decimal=deci)
                    except AssertionError as ex:
                        raise AssertionError(
                            "Output %d of test %d in folder %r failed."
                            % (i, index, self.folder)
                        ) from ex
            elif hasattr(o, "is_compatible"):
                # A shape
                if e.dtype != o.dtype:
                    raise AssertionError(
                        "Output %d of test %d in folder %r failed "
                        "(e.dtype=%r, o=%r)." % (i, index, self.folder, e.dtype, o)
                    )
                if not o.is_compatible(e.shape):
                    raise AssertionError(  # pragma: no cover
                        "Output %d of test %d in folder %r failed "
                        "(e.shape=%r, o=%r)." % (i, index, self.folder, e.shape, o)
                    )
        else:
            raise NotImplementedError(
                "Comparison not implemented for type %r." % type(e)
            )

    def is_random(self):
        "Tells if a test is random or not."
        if "bernoulli" in self.folder:
            return True
        return False

    def run(self, load_fct, run_fct, index=None, decimal=None):
        """
        Executes a tests or all tests if index is None.
        The function crashes if the tests fails.

        :param load_fct: loading function, takes a loaded onnx graph,
            and returns an object
        :param run_fct: running function, takes the result of previous
            function, the inputs, and returns the outputs
        :param index: index of the test to run or all.
        :param decimal: requested precision to compare results
        """
        if index is None:
            for i in range(len(self)):
                self.run(load_fct, run_fct, index=i, decimal=decimal)
            return

        obj = load_fct(self.onnx_model)

        got = run_fct(obj, *self.tests[index]["inputs"])
        expected = self.tests[index]["outputs"]
        if len(got) != len(expected):
            raise AssertionError(  # pragma: no cover
                "Unexpected number of output (test %d, folder %r), "
                "got %r, expected %r." % (index, self.folder, len(got), len(expected))
            )
        for i, (e, o) in enumerate(zip(expected, got)):
            if self.is_random():
                if e.dtype != o.dtype:
                    raise AssertionError(
                        "Output %d of test %d in folder %r failed "
                        "(type mismatch %r != %r)."
                        % (i, index, self.folder, e.dtype, o.dtype)
                    )
                if e.shape != o.shape:
                    raise AssertionError(
                        "Output %d of test %d in folder %r failed "
                        "(shape mismatch %r != %r)."
                        % (i, index, self.folder, e.shape, o.shape)
                    )
            else:
                self._compare_results(index, i, e, o, decimal=decimal)

    def to_python(self):
        """
        Returns a python code equivalent to the ONNX test.

        :return: code
        """
        from ..onnx_tools.onnx_export import export2onnx

        rows = []
        code = export2onnx(self.onnx_model)
        lines = code.split("\n")
        lines = [
            line
            for line in lines
            if not line.strip().startswith("print")
            and not line.strip().startswith("# ")
        ]
        rows.append(textwrap.dedent("\n".join(lines)))
        rows.append("oinf = OnnxInference(onnx_model)")
        for test in self.tests:
            rows.append("xs = [")
            for inp in test["inputs"]:
                rows.append(textwrap.indent(repr(inp) + ",", "    " * 2))
            rows.append("]")
            rows.append("ys = [")
            for out in test["outputs"]:
                rows.append(textwrap.indent(repr(out) + ",", "    " * 2))
            rows.append("]")
            rows.append("feeds = {n: x for n, x in zip(oinf.input_names, xs)}")
            rows.append("got = oinf.run(feeds)")
            rows.append("goty = [got[k] for k in oinf.output_names]")
            rows.append("for y, gy in zip(ys, goty):")
            rows.append("    self.assertEqualArray(y, gy)")
            rows.append("")
        code = "\n".join(rows)
        final = "\n".join(["def %s(self):" % self.name, textwrap.indent(code, "    ")])
        try:
            from pyquickhelper.pycode.code_helper import remove_extra_spaces_and_pep8
        except ImportError:  # pragma: no cover
            return final
        return remove_extra_spaces_and_pep8(final, aggressive=True)


def enumerate_onnx_tests(series, fct_filter=None):
    """
    Collects test from a sub folder of `onnx/backend/test`.
    Works as an enumerator to start processing them
    without waiting or storing too much of them.

    :param series: which subfolder to load, possible values:
        (`'node'`, ...)
    :param fct_filter: function `lambda testname: boolean`
        to load or skip the test, None for all
    :return: list of @see cl OnnxBackendTest
    """
    root = os.path.dirname(backend_folder)
    sub = os.path.join(root, "data", series)
    if not os.path.exists(sub):
        raise FileNotFoundError(
            "Unable to find series of tests in %r, subfolders:\n%s"
            % (root, "\n".join(os.listdir(root)))
        )
    tests = os.listdir(sub)
    for t in tests:
        if fct_filter is not None and not fct_filter(t):
            continue
        folder = os.path.join(sub, t)
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {".onnx"}]
        if len(onx) == 1:
            yield OnnxBackendTest(folder)
