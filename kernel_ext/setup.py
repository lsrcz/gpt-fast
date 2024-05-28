from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="kernel_ext",
    packages=["kernel_ext", "kernel_ext-stubs"],
    include_package_data=True,
    ext_modules=[
        cpp_extension.CUDAExtension(  # type: ignore
            "kernel_ext",
            [
                "kernel_ext/kernel.cu",
            ],
            extra_compile_args={"nvcc": ["--extended-lambda"]},
            include_dirs=["kernel_ext"],
        ),
    ],
    setup_requires=["pybind11-stubgen"],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    package_data={"kernel_ext-stubs": ["*.pyi"]},
)
