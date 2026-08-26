# Transformer GPU Optimization Benchmark

The PyTorch benchmark compares the explicit reference Transformer with an
optimized implementation that uses a fused QKV projection and PyTorch scaled
dot-product attention. On CUDA, SDPA can dispatch to fused Flash Attention.

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
python torch_transformer_benchmark.py --device cuda --dtype float32 \
  --batch-size 8 --seq-len 128

python torch_transformer_benchmark.py --device cuda --dtype float32 \
  --causal --padding-ratio 0.35
```

Every performance run reports median, mean, p90 and minimum latency, throughput,
speedup, and peak CUDA activation allocation.
