"""Test automation with nox. https://nox.thea.codes/"""

import nox


@nox.session()
def build(session):
    """Build package."""
    session.install("build", "wheel")
    session.run("python", "-m", "build")


@nox.session()
def test_onnx_func_expe(session):
    """test with onnx function experiment builds"""
    session.install(
        "autopep8",
        "click",
        "jinja2",
        "numpy==1.23.5",
        "protobuf<4",
        "typing_extensions",
        "beartype",
        "types-PyYAML",
        "expecttest",
        "packaging",
        "parameterized",
        "pytest-cov",
        "pytest-subtests",
        "pytest-xdist",
        "pytest!=7.1.0",
        "pyyaml",
        "ort-function-experiment-nightly",
        "torch==1.13",
    )
    session.install(".")
    session.run("python", "-I", "-m", "pip", "uninstall", "-y", "onnx")
    session.run(
        "python",
        "-I",
        "-m",
        "pip",
        "install",
        "-f",
        "https://onnxruntimepackages.z14.web.core.windows.net/onnx-function-experiment.html",
        "--pre",
        "onnx-function-experiment",
    )
    session.run("pip", "list")
    session.run("pytest", "onnxscript")
    session.run("pytest", "docs/test")


@nox.session()
def test_onnx_weekly(session):
    """test with onnx main branch"""
    session.install(
        "autopep8",
        "click",
        "jinja2",
        "numpy==1.23.5",
        "protobuf<4",
        "typing_extensions",
        "beartype",
        "types-PyYAML",
        "expecttest",
        "packaging",
        "parameterized",
        "pytest-cov",
        "pytest-subtests",
        "pytest-xdist",
        "pytest!=7.1.0",
        "pyyaml",
        "onnx-weekly",
        "onnxruntime",
        "torch==1.13",
    )
    session.install(".")
    session.run("pip", "uninstall", "onnx", "-y")
    session.run("pip", "list")
    session.run("pytest", "onnxscript")


@nox.session()
def test_torch_nightly(session):
    """test with pytorch nightly"""
    session.install(
        "autopep8",
        "click",
        "jinja2",
        "numpy==1.23.5",
        "protobuf<4",
        "typing_extensions",
        "beartype",
        "types-PyYAML",
        "expecttest",
        "packaging",
        "parameterized",
        "pytest-cov",
        "pytest-subtests",
        "pytest-xdist",
        "pytest!=7.1.0",
        "pyyaml",
        "onnx==1.13",
        "onnxruntime",
        "torch",
    )
    session.install(".")
    session.run("pip", "list")
    session.run("pytest", "onnxscript")
