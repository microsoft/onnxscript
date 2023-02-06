"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import nox

nox.options.error_on_missing_interpreters = False


COMMON_TEST_DEPENDENCIES = (
    "autopep8",
    "click",
    "jinja2",
    'numpy==1.23.5; python_version>="3.8"',
    'numpy; python_version<"3.8"',
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
)
ONNX = "onnx==1.13"
ONNX_RUNTIME = "onnxruntime==1.13.1"
PYTORCH = "torch==1.13"


@nox.session(tags=["build"])
def build(session):
    """Build package."""
    session.install("build", "wheel")
    session.run("python", "-m", "build")


@nox.session(tags=["test"])
def test(session):
    """Test onnxscript and documentation."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
        ONNX,
        ONNX_RUNTIME,
    )
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)
    session.run("pytest", "docs/test", *session.posargs)


@nox.session(tags=["test-function-experiment"])
def test_onnx_func_expe(session):
    """Test with onnx function experiment builds."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
    )
    # Install ONNX and ORT with experimental ONNX function support
    session.install(
        "-f",
        "https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime-function-experiment.html",
        "--pre",
        "ort-function-experiment-nightly",
    )
    session.install(
        "-f",
        "https://onnxruntimepackages.z14.web.core.windows.net/onnx-function-experiment.html",
        "--pre",
        "onnx-function-experiment",
    )
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)
    session.run("pytest", "docs/test", *session.posargs)


@nox.session(tags=["test-torch-nightly"])
def test_torch_nightly(session):
    """Test with PyTorch nightly (preview) build."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        ONNX,
        ONNX_RUNTIME,
    )
    session.install(
        "--pre", "torch", "--index-url", "https://download.pytorch.org/whl/nightly/cpu"
    )
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)


@nox.session(tags=["test-onnx-weekly"])
def test_onnx_weekly(session):
    """Test with ONNX weekly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES, ONNX_RUNTIME, PYTORCH, "wheel")
    session.install("--index-url", "https://test.pypi.org/simple/", "onnx-weekly")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)
