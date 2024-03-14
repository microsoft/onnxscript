"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import nox

nox.options.error_on_missing_interpreters = False


COMMON_TEST_DEPENDENCIES = (
    "jinja2",
    "numpy==1.24.4",
    "typing_extensions",
    "beartype!=0.16.0",
    "types-PyYAML",
    "expecttest==0.1.6",
    "hypothesis",
    "packaging",
    "parameterized",
    "pytest-cov",
    "pytest-randomly",
    "pytest-subtests",
    "pytest-xdist",
    "pytest!=7.1.0",
    "pyyaml",
)
ONNX = "onnx==1.15.0"
ONNX_RUNTIME = "onnxruntime==1.17.1"
PYTORCH = "torch==2.1.0"
TORCHVISON = "torchvision==0.16"
ONNX_RUNTIME_NIGHTLY_DEPENDENCIES = (
    "flatbuffers",
    "coloredlogs",
    "sympy",
    "numpy",
    "packaging",
    "protobuf",
)


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
        TORCHVISON,
        ONNX,
        ONNX_RUNTIME,
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
        ONNX_RUNTIME,
    )
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install("-r", "requirements/ci/requirements-pytorch-nightly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)


@nox.session(tags=["test-onnx-weekly"])
def test_onnx_weekly(session):
    """Test with ONNX weekly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES, ONNX_RUNTIME, PYTORCH, TORCHVISON)
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)


@nox.session(tags=["test-ort-nightly"])
def test_ort_nightly(session):
    """Test with ONNX Runtime nightly builds."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
        TORCHVISON,
        ONNX,
        *ONNX_RUNTIME_NIGHTLY_DEPENDENCIES,
    )
    session.install("-r", "requirements/ci/requirements-ort-nightly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", *session.posargs)


@nox.session(tags=["test-experimental-torchlib-tracing"])
def test_experimental_torchlib_tracing(session):
    """Test TorchLib with the experimental TORCHLIB_EXPERIMENTAL_PREFER_TRACING flag on."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
        TORCHVISON,
        ONNX,
        *ONNX_RUNTIME_NIGHTLY_DEPENDENCIES,
    )
    session.install("-r", "requirements/ci/requirements-ort-nightly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run(
        "pytest",
        "onnxscript/tests/function_libs/torch_lib/ops_test.py",
        *session.posargs,
        env={"TORCHLIB_EXPERIMENTAL_PREFER_TRACING": "1"},
    )
