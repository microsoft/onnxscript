# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import os
import textwrap
from typing import Iterator

import numpy as np
import onnx
import onnx.numpy_helper
from onnx.backend.test import __file__ as backend_folder

from onnxscript.backend import onnx_export


def assert_almost_equal_string(expected, value):
    """Compares two arrays knowing they contain strings.
    Raises an exception if the test fails.

    Args:
        expected: expected array
        value: value
    """

    def is_float(x):  # pylint: disable=unused-argument
        try:
            return True
        except ValueError:  # pragma: no cover
            return False

    if all(map(is_float, expected.ravel())):
        expected_float = expected.astype(np.float32)
        value_float = value.astype(np.float32)
        np.testing.assert_almost_equal(expected_float, value_float)
    else:
        np.testing.assert_almost_equal(expected, value)


class OnnxBackendTest:
    """Definition of a backend test. It starts with a folder,
    in this folder, one onnx file must be there, then a subfolder
    for each test to run with this model.

    Args:
        folder: test folder
        onnx_path: onnx file
        onnx_model: loaded onnx file
        tests: list of test
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
            raise FileNotFoundError(f"File not found: {full!r}.")  # pragma: no cover
        with open(full, "rb") as f:
            serialized = f.read()
        try:
            loaded = onnx.numpy_helper.to_array(onnx.load_tensor_from_string(serialized))
        except Exception as e:  # pylint: disable=W0703
            seq = onnx.SequenceProto()
            try:
                seq.ParseFromString(serialized)
                loaded = onnx.numpy_helper.to_list(seq)  # type: ignore[assignment]
            except Exception:  # pylint: disable=W0703
                try:
                    loaded = onnx.load_model_from_string(serialized)  # type: ignore[assignment]
                except Exception:
                    raise RuntimeError(
                        f"Unable to read {full!r}, error is {e}, "
                        f"content is {serialized[:100]!r}."
                    ) from e
        return loaded

    @staticmethod
    def _load(folder, names):
        res = []
        for name in names:
            full = os.path.join(folder, name)
            new_tensor = OnnxBackendTest._read_proto_from_file(full)
            if isinstance(new_tensor, (np.ndarray, onnx.ModelProto, list)):
                t = new_tensor
            elif isinstance(new_tensor, onnx.TensorProto):
                t = onnx.numpy_helper.to_array(new_tensor)
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Unexpected type {type(new_tensor)!r} for {full!r}."
                )
            res.append(t)
        return res

    def __repr__(self):
        """Usual"""
        return f"{self.__class__.__name__}({self.folder!r})"

    def __init__(self, folder):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Unable to find folder {folder!r}.")  # pragma: no cover
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {".onnx"}]
        if len(onx) != 1:
            raise ValueError(  # pragma: no cover
                f"There is more than one onnx file in {folder!r} ({onx!r})."
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
                outputs = OnnxBackendTest._sort(c for c in pb if c.startswith("output_"))

                t = dict(
                    inputs=OnnxBackendTest._load(full, inputs),
                    outputs=OnnxBackendTest._load(full, outputs),
                )
                self.tests.append(t)

    @property
    def name(self):
        """Returns the test name."""
        return os.path.split(self.folder)[-1]

    def __len__(self):
        """Returns the number of tests."""
        return len(self.tests)

    def _compare_results(self, index, i, e, o, decimal=None):
        """Compares the expected output and the output produced
        by the runtime. Raises an exception if not equal.

        Args:
            index: test index
            i: output index
            e: expected output
            o: output
            decimal: precision
        """
        if isinstance(e, np.ndarray):
            if isinstance(o, np.ndarray):
                if decimal is None:
                    if e.dtype == np.float32:
                        deci = 6
                    elif e.dtype == np.float64:
                        deci = 12
                    else:
                        deci = 7
                else:
                    deci = decimal
                if e.dtype == np.object_:
                    try:
                        assert_almost_equal_string(e, o)
                    except AssertionError as ex:
                        raise AssertionError(  # pragma: no cover
                            f"Output {i} of test {index} in folder {self.folder} failed."
                        ) from ex
                else:
                    try:
                        np.testing.assert_almost_equal(e, o, decimal=deci)
                    except AssertionError as ex:
                        raise AssertionError(
                            f"Output {i} of test {index} in folder {self.folder} failed."
                        ) from ex
            elif hasattr(o, "is_compatible"):
                # A shape
                if e.dtype != o.dtype:
                    raise AssertionError(
                        f"Output {i} of test {index} in folder "
                        f"{self.folder} failed (e.dtype={e.dtype}, o={o})."
                    )
                if not o.is_compatible(e.shape):
                    raise AssertionError(  # pragma: no cover
                        f"Output {i} of test {index} in folder "
                        f"{self.folder} failed (e.shape={e.shape}, o={o})."
                    )
        else:
            raise NotImplementedError(f"Comparison not implemented for type {type(e)!r}.")

    def is_random(self):
        """Returns whether the test is random."""
        return "bernoulli" in self.folder

    def run(self, load_fct, run_fct, index=None, decimal=None):
        """Executes a tests or all tests if index is None.
        The function crashes if the tests fails.

        Args:
            load_fct: loading function, takes a loaded onnx graph, and
                returns an object
            run_fct: running function, takes the result of previous
                function, the inputs, and returns the outputs
            index: index of the test to run or all.
            decimal: requested precision to compare results
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
                f"Unexpected number of output (test {index}, folder {self.folder}), "
                f"got {len(got)}, expected {len(expected)}."
            )
        for i, (e, o) in enumerate(zip(expected, got)):
            if self.is_random():
                if e.dtype != o.dtype:
                    raise AssertionError(
                        f"Output {i} of test {index} in folder "
                        f"{self.folder} failed (type mismatch {e.dtype} != {o.dtype})."
                    )
                if e.shape != o.shape:
                    raise AssertionError(
                        f"Output {i} of test {index} in folder "
                        f"{self.folder} failed (shape mismatch {e.shape} != {o.shape})."
                    )
            else:
                self._compare_results(index, i, e, o, decimal=decimal)

    def to_python(self):
        """Returns a python code equivalent to the ONNX test.

        Returns:
            code
        """
        rows = []
        code = onnx_export.export2onnx(self.onnx_model)  # type: ignore[attr-defined]
        lines = code.split("\n")
        lines = [
            line
            for line in lines
            if not line.strip().startswith("print") and not line.strip().startswith("# ")
        ]
        rows.append(textwrap.dedent("\n".join(lines)))
        rows.append("oinf = OnnxInference(onnx_model)")
        for test in self.tests:
            rows.append("xs = [")
            for inp in test["inputs"]:
                rows.append(textwrap.indent(f"{inp!r},", "    " * 2))
            rows.append("]")
            rows.append("ys = [")
            for out in test["outputs"]:
                rows.append(textwrap.indent(f"{out!r},", "    " * 2))
            rows.append("]")
            rows.append("feeds = {n: x for n, x in zip(oinf.input_names, xs)}")
            rows.append("got = oinf.run(feeds)")
            rows.append("goty = [got[k] for k in oinf.output_names]")
            rows.append("for y, gy in zip(ys, goty):")
            rows.append("    self.assertEqualArray(y, gy)")
            rows.append("")
        code = "\n".join(rows)
        final = "\n".join([f"def {self.name}(self):", textwrap.indent(code, "    ")])
        return final


def enumerate_onnx_tests(series, fct_filter=None) -> Iterator[OnnxBackendTest]:
    """Collects test from a sub folder of `onnx/backend/test`.
    Works as an enumerator to start processing them
    without waiting or storing too much of them.

    Args:
        series: which subfolder to load, possible values: (`'node'`,
            ...)
        fct_filter: function `lambda testname: boolean` to load or skip
            the test, None for all

    Yields:
        list of @see cl OnnxBackendTest
    """
    root = os.path.dirname(backend_folder)
    sub = os.path.join(root, "data", series)
    if not os.path.exists(sub):
        raise FileNotFoundError(
            "Unable to find series of tests in {root!r}, subfolders:\n"
            + "\n".join(os.listdir(root))
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
