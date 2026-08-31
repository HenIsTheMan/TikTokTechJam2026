#!/usr/bin/env python3
"""Build the shape-specialized CUDA extension for the RTX 5090."""

from __future__ import annotations

import os
from pathlib import Path

# RTX 5090 = Blackwell, compute capability 12.0.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="transformer_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="transformer_cuda_ext",
            sources=[
                str(ROOT / "cuda_extension" / "binding.cpp"),
                str(ROOT / "cuda_extension" / "kernels.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-DNDEBUG"],
                "nvcc": [
                    "-O3",
                    "-DNDEBUG",
                    "-arch=sm_120",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
