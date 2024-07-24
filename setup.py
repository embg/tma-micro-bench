from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tma_kernels",
    ext_modules=[
        CUDAExtension(
            "tma_kernels",
            [
                "tma_kernels.cpp",
                "tma_kernels_cuda.cu",
            ],
            libraries=["cuda"]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
