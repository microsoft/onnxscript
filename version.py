# torch-ort 1.13.0.dev20221022
# onnx_function_experiment-1.12.0.dev20220629-cp310-cp310-win32.whl
# onnxruntime-1.8.0.dev202104084+cu111.training-cp37-cp37m-manylinux2014_x86_64.whl
from setuptools_scm import ScmVersion

def get_version(version: ScmVersion):
    import pdb; pdb.set_trace()
    from setuptools_scm.version import guess_next_version
    return version.format_next_version(guess_next_version, '{guessed}b{distance}')
