# Transformer GPU Optimization Benchmark

The PyTorch benchmark compares the explicit reference Transformer with an
optimized implementation that uses a fused QKV projection and PyTorch scaled
dot-product attention. On CUDA, SDPA can dispatch to fused Flash Attention.

## Setup
```bash
python3 -m venv .venv
source venv/bin/activate
pip install numpy torch
```

## Run the accuracy-safe CUDA path

```bash
python torch_transformer_benchmark.py \
  --device cuda \
  --dtype float32 \
  --compile-user \
  --compile-mode reduce-overhead
```

TF32 is disabled by default because its rounding can exceed the benchmark's
strict per-element tolerance. It can be tested explicitly with `--allow-tf32`.

## Build and compare the custom RTX 5090 backend

The custom extension fuses residual addition, projection bias, padding-mask
handling and LayerNorm, and provides a vectorized exact-GELU epilogue. Matrix
multiplication remains on cuBLAS/cuBLASLt and attention remains on fused SDPA;
a handwritten GEMM is not competitive with those libraries for this workload.

```bash
make build-cuda
make benchmark-all
```

The build defaults to `/usr/local/cuda` and `TORCH_CUDA_ARCH_LIST=12.0`. Override
either Make variable if needed. Generated binaries and build products are
ignored by Git.

Select a backend explicitly with `--optimized-backend pytorch`,
`--optimized-backend cuda`, or `--optimized-backend cuda-hybrid`. The CUDA
backends are intentionally limited to the default CUDA float32 shape. The
validated hybrid uses TF32 tensor cores only for all FFN contraction GEMMs and
the first three FFN expansion GEMMs; attention and the remaining projections stay
strict FP32. `auto` selects this hybrid when the extension is available and
falls back to compiled PyTorch otherwise.

Representative RTX 5090 validation for the default FP32 shape:

| Backend | Median latency | Baseline speedup | Strict accuracy |
| --- | ---: | ---: | --- |
| Compiled PyTorch | 1.2036 ms | 1.880x | PASS |
| Custom CUDA | 1.1986 ms | 1.889x | PASS |
| CUDA hybrid | 0.9323 ms | 2.268x | PASS |

The strict custom result is approximately 0.4% faster than compiled PyTorch.
The selective-TF32 hybrid is about 23% faster and cleared the 3% promotion
gate. It also passed the causal 35%-padding accuracy suite and reached a 2.520x
baseline speedup. Full-model TF32 remains disabled because it fails that suite.

BF16 and FP16 can be benchmarked with `--dtype bfloat16` and `--dtype float16`.
For this six-layer reference, fused attention's normal reduced-precision
rounding differences may exceed the default `atol=0.001` / `rtol=0.01`; the
script reports and stops on those failures unless `--benchmark-on-failure` is
specified.

## Useful shape tests

```bash
python Sandbox.py --device cuda --dtype float32 \
  --batch-size 8 --seq-len 128

python Sandbox.py --device cuda --dtype float32 \
  --causal --padding-ratio 0.35
```

Every performance run reports median, mean, p90 and minimum latency, throughput,
speedup, and peak CUDA activation allocation.

## Project description

This project optimizes the inference runtime of a Transformer neural network on
an NVIDIA GPU while preserving numerical correctness relative to a reference
PyTorch implementation. The benchmark models a six-layer Transformer with a
model width of 512, eight attention heads, a feed-forward width of 2,048, a
batch size of eight, and a sequence length of 128.

Each Transformer block follows a pre-normalization pipeline:

1. Layer normalization
2. Multi-head self-attention
3. Residual addition
4. Layer normalization
5. Two-layer feed-forward network with exact GELU activation
6. Residual addition

A final layer normalization produces the output. The implementation also
supports causal attention and variable-length sequences represented by padding
masks.

## How the solution addresses the problem statement

The problem asks for a faster Transformer implementation on the target GPU
without sacrificing numerical correctness. This solution provides an explicit
PyTorch baseline and several optimized backends that use identical model
weights, inputs, and output shapes for a fair comparison.

The main optimizations are:

- **Fused QKV projection:** The three separate query, key, and value linear
  projections are combined into one larger projection, reducing operator and
  kernel-launch overhead.
- **Fused scaled dot-product attention:** PyTorch's
  `scaled_dot_product_attention` operation replaces the manually assembled
  score, mask, softmax, and value-multiplication pipeline. On supported CUDA
  hardware, PyTorch can dispatch this operation to a fused attention kernel.
- **Fused residual and LayerNorm CUDA kernel:** A custom, shape-specialized
  kernel combines projection bias, residual addition, padding-mask handling,
  LayerNorm statistics, the affine LayerNorm transform, and output writes. It
  uses one CUDA block per token row, `float4` vectorized memory operations, and
  warp-level reductions to reduce global-memory traffic and intermediate
  allocations.
- **Fused bias and exact-GELU CUDA kernel:** The feed-forward expansion bias and
  exact GELU activation are evaluated together using vectorized CUDA loads and
  stores.
- **Optimized matrix multiplication:** Attention and feed-forward matrix
  multiplications remain on PyTorch's tuned CUDA paths backed by NVIDIA
  libraries. The project focuses custom kernels on the memory-bound epilogues
  instead of replacing highly optimized GEMM implementations.
- **Selective TF32:** The hybrid backend enables TF32 tensor-core execution for
  all feed-forward contraction GEMMs and only the first three feed-forward
  expansion GEMMs. Attention projections and the remaining operations stay in
  strict FP32 because they are more sensitive to accumulated rounding error.
- **Ahead-of-time CUDA extension:** Custom kernels are compiled before the
  benchmark, avoiding runtime compilation during measured inference.
- **Optional `torch.compile`:** The portable optimized PyTorch backend can be
  compiled in `reduce-overhead` mode to reduce Python and launch overhead.

Correctness is checked before performance is reported. Every output element
must satisfy either an absolute error of at most `0.001` or a relative error of
at most `1%`. The test uses several deterministic random trials and reports the
number of failures, maximum absolute and relative errors, and the worst output
location. Unless explicitly overridden, benchmarking stops if validation
fails.

For performance testing, the program warms up both implementations and then
uses CUDA Events to collect repeated latency samples from a fixed input. It
alternates the measurement order to reduce thermal and clock-order bias and
reports median, mean, p90, and minimum latency, token throughput, speedup, and
peak CUDA activation allocation.

Representative RTX 5090 validation for the default FP32 workload produced the
following results:

| Backend | Median latency | Baseline speedup | Strict accuracy |
| --- | ---: | ---: | --- |
| Compiled optimized PyTorch | 1.2036 ms | 1.880x | PASS |
| Custom CUDA | 1.1986 ms | 1.889x | PASS |
| Selective-TF32 CUDA hybrid | 0.9323 ms | 2.268x | PASS |

The hybrid backend was approximately 23% faster than the compiled optimized
PyTorch implementation. It also passed the causal test with 35% padding and
reached a 2.520x speedup over the corresponding baseline. Full-model TF32 was
not used because it failed this stricter accuracy case.

## Development tools used

- **Visual Studio Code:** Used to inspect, edit, and organize the Python,
  C++, CUDA, Markdown, and build files.
- **OpenAI Codex:** Used as an AI-assisted development tool for implementation
  analysis, documentation, and presentation generation.
- **Python virtual environment and pip:** Used to isolate Python dependencies.
- **GNU Make:** Provides repeatable commands for building the extension and
  running each benchmark backend.
- **NVIDIA CUDA Toolkit and NVCC:** Used to compile the custom CUDA kernels for
  the target GPU architecture.
- **Git:** Used for source control and change tracking.

The project was developed as local source code rather than in Google Colab or
Jupyter Notebook.

## APIs used

The benchmark does not call any external web, mapping, data, or hosted AI API
at runtime. It uses local framework APIs, including:

- **PyTorch Python API:** Model construction, tensor operations, inference,
  scaled dot-product attention, CUDA Events, memory statistics, and optional
  graph compilation.
- **PyTorch C++/CUDA Extension API:** Exposes the custom CUDA kernels to Python
  through `CUDAExtension`, ATen, and pybind11 bindings.
- **CUDA Runtime API:** Kernel launches, CUDA streams, device guards, and launch
  error checking.

OpenAI Codex assisted development, but the finished benchmark does not depend
on an OpenAI API or network connection when it runs.

## Libraries and frameworks used

- **PyTorch:** Implements the baseline and optimized Transformer models and
  supplies tensor, neural-network, SDPA, compilation, and benchmarking
  functionality.
- **CUDA, ATen, and PyTorch C++ extensions:** Implement and integrate the custom
  GPU kernels.
- **cuBLAS/cuBLASLt through PyTorch:** Execute the optimized projection and
  feed-forward matrix multiplications.
- **Python standard library:** `argparse`, `copy`, `dataclasses`, `math`,
  `statistics`, and `time` support configuration, validation, and reporting.
- **setuptools:** Builds the ahead-of-time CUDA extension.

Hugging Face Transformers, TensorFlow, scikit-learn, pandas, and NumPy are not
used by the benchmark implementation.

## Datasets and assets used

No external dataset, pretrained model, private user data, or manually labelled
data is required.

The benchmark generates synthetic input tensors with `torch.randn` using fixed
random seeds for reproducibility. When padding tests are enabled, token lengths
and boolean validity masks are also generated deterministically. The baseline
model is initialized locally, and its exact weights are copied into the
optimized model; therefore, the comparison measures implementation differences
rather than differences between trained models.

Project assets consist of the source code, custom CUDA kernels, build scripts,
the documented representative RTX 5090 benchmark measurements, and the
generated PowerPoint presentation. The TensorFlow benchmark is outside the
scope of this submission; all implementation and evaluation work focuses on
`torch_transformer_benchmark.py`.
