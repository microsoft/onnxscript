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
    'numpy==1.24.4; python_version<"3.9"',
    'numpy==1.26.4; python_version>="3.9"',
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
    "typing_extensions",
    "ml-dtypes",
)
ONNX = "onnx==1.16"
ONNX_RUNTIME = "onnxruntime==1.17.1"
PYTORCH = "torch==2.2.2"
TORCHVISON = "torchvision==0.17.2"
TRANSFORMERS = "transformers==4.37.2"
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
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test-onnx-weekly"])
def test_onnx_weekly(session):
    """Test with ONNX weekly (preview) build."""
    session.install(*COMMON_TEST_DEPENDENCIES, ONNX_RUNTIME, PYTORCH, TORCHVISON, TRANSFORMERS)
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
        *ONNX_RUNTIME_NIGHTLY_DEPENDENCIES,
    )
    session.install("-r", "requirements/ci/requirements-ort-nightly.txt")
    session.install(".", "--no-deps")
    session.run("pip", "list")
    session.run("pytest", "onnxscript", "--doctest-modules", *session.posargs)
    session.run("pytest", "tests", *session.posargs)


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
        "tests/function_libs/torch_lib/ops_test.py",
        *session.posargs,
        env={"TORCHLIB_EXPERIMENTAL_PREFER_TRACING": "1"},
    )


@nox.session(tags=["test-experimental-torchlib-onnx-ir"])
def test_experimental_torchlib_onnx_ir(session):
    """Test TorchLib using the ONNX IR to build graphs."""
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
        "tests/function_libs/torch_lib/ops_test.py",
        *session.posargs,
        env={"TORCHLIB_EXPERIMENTAL_USE_IR": "1"},
    )


@nox.session(tags=["test-dort"])
def test_dort(session):
    """Test the conversion of a couple of models from transformers."""
    session.install(
        *COMMON_TEST_DEPENDENCIES,
    )
    torch_version, transformers_version = session.posargs

    if torch_version == "nightly":
        session.install(
            "--pre",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cpu",
        )
    else:
        session.install("torch", "torchvision", "torchaudio")

    session.install("torch", "torchvision", "torchaudio")
    session.install(f"transformers=={transformers_version}")
    session.install("onnxruntime-training==1.17.1")

    session.run("pip", "list")
    session.run("pytest", "onnxscript")
    session.run("pytest", "tests")
