# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import nox

nox.options.error_on_missing_interpreters = False


COMMON_TEST_DEPENDENCIES = (
    "beartype==0.17.2",
    "expecttest==0.1.6",
    "hypothesis",
    "numpy",
    "packaging",
    "parameterized",
    'psutil; sys_platform != "win32"',
    "pytest-cov",
    "pytest-randomly",
    "pytest-subtests",
    "pytest-xdist",
    "pytest!=7.1.0",
    "pyyaml",
    "types-PyYAML",
    "typing_extensions>=4.10",
    "ml-dtypes",
)
ONNX = "onnx==1.17"
ONNX_RUNTIME = "onnxruntime==1.20.1"
PYTORCH = "torch==2.5.1"
TORCHVISON = "torchvision==0.20.1"
TRANSFORMERS = "transformers==4.37.2"
ONNX_RUNTIME_NIGHTLY_DEPENDENCIES = (
    "flatbuffers",
    "coloredlogs",
    "sympy",
    "numpy",
    "packaging",
    "protobuf",
)
ONNX_IR = "onnx_ir==0.1.3"
ONNX_IR_MAIN = "git+https://github.com/onnx/ir-py.git@main#egg=onnx_ir"


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
        ONNX_IR,
        ONNX_RUNTIME,
        TRANSFORMERS,
    )
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", "docs/test", *session.posargs)


@nox.session(tags=["test-torch-nightly"])
def test_torch_nightly(session):
    """Test with PyTorch nightly (preview) build."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        ONNX_RUNTIME,
        TRANSFORMERS,
    )
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install("-r", "requirements/ci/requirements-pytorch-nightly.txt")
    session.install(ONNX_IR, "--no-deps")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-onnx-weekly"])
def test_onnx_weekly(session):
    """Test with ONNX weekly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES, ONNX_RUNTIME, PYTORCH, TORCHVISON, TRANSFORMERS)
    session.install(ONNX_IR, "--no-deps")
    session.install("-r", "requirements/ci/requirements-onnx-weekly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-ort-nightly"])
def test_ort_nightly(session):
    """Test with ONNX Runtime nightly builds."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
        TORCHVISON,
        ONNX,
        ONNX_IR,
        TRANSFORMERS,
        *ONNX_RUNTIME_NIGHTLY_DEPENDENCIES,
    )
    session.install("-r", "requirements/ci/requirements-ort-nightly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-onnx-ir-git"])
def test_onnx_ir_git(session):
    """Test with ONNX IR Git builds."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
        PYTORCH,
        TORCHVISON,
        ONNX,
        ONNX_RUNTIME,
        TRANSFORMERS,
    )
    session.install(ONNX_IR_MAIN)
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)
