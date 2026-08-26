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
