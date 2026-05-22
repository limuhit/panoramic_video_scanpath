#!/usr/bin/env python3
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


extension_sources = [
    "extension/main.cpp",
    "extension/math_cuda.cu",
    "extension/projects_cuda.cu",
    "extension/dtow_cuda.cu",
    "extension/viewport_cuda.cu",
    "extension/InvTransSample_cuda.cu",
    "extension/erp2vp_cuda.cu",
    "extension/GMM_2D_Table_cuda.cu",
    "extension/vp2erp_cuda.cu",
    "extension/linear_mask_cuda.cu",
    "extension/pre_data_cuda.cu",
    "extension/viewport_batch_cuda.cu",
    "extension/data_manager_cuda.cu",
    "extension/gmm_sample_cuda.cu",
    "extension/viewport_batch_eval_cuda.cu",
]


setup(
    name="spath",
    version="1.0.0",
    description="Core training and inference code for panoramic video scanpath prediction.",
    packages=find_packages(include=["spath", "spath.*", "SPath_operator", "SPath_operator.*"]),
    ext_modules=[
        CUDAExtension(
            "SPath",
            extension_sources,
            include_dirs=["extension"],
            extra_compile_args={
                "cxx": ["-std=c++14", "-DOK"],
                "nvcc": ["-D__CUDA_NO_HALF_OPERATORS__"],
            },
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
