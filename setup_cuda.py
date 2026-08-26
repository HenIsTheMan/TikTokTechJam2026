#!/usr/bin/env python3
"""Build the RTX 5090 Transformer CUDA extension in place."""

from __future__ import annotations

import os
from pathlib import Path

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
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
