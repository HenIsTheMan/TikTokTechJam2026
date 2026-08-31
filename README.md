# TikTokTechJam2026

<h3>Project Name: Optimus Time</h3>

<h3>Team Name: bench-5</h3>

## Project Overview

This project benchmarks an inference-only PyTorch Transformer against a UserOptimizedTransformer.

The UserOptimizedTransformer uses explicit multi-head attention, LayerNorm, residual connections, and an exact GELU FFN
while preserving the same parameter values and keeping within the stipulated error constraints.

The UserOptimizedTransformer provides a CPU non-CUDA implementation and a GPU CUDA implementation for users to choose from,
both providing substantial speedups.

## Installation and Setup

<h4>Run the following commands in the order they are listed:</h4>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy ninja torch
```

## Reproducing Results

In the demo video, we used a machine with a AMD Ryzen 9 9950X as the CPU and an NVIDIA RTX 5090 as the GPU.
But the program can be run on any machine that has basic compute capabilities.

<h4>The following build and run commands are available:</h4>

```bash
[Runs CPU non-CUDA implementation] make
[Runs CPU non-CUDA implementation] make cpu
[Runs GPU CUDA implementation] make cuda
[Removes all build files] make clean
```

<h4>These build and run commands by default executes the default configuration and workload shown below:</h4>

```text
batch_size=8
seq_len=128
d_model=512
heads=8
ffn_dim=2048
layers=6
causal=False

device="auto"
dtype="float32"
padding_ratio=0.0
input_scale=1.0

accuracy_trials=5
rtol=0.01
atol=0.001
seed=1234

warmup=7
repeats=5
benchmark_rounds=2
benchmark_on_failure=False

compile_baseline=False
compile_user=False
compile_mode="default"
non_strict_weight_copy=False
matmul_precision="high"
allow_tf32=True
```

For a customized workload, users can modify the input parameters with command-line arguments.

<h4>An example of customizing the input parameters with command-line arguments is shown below:</h4>

```bash
make CMD_LINE_ARGS="--batch-size 10 --seq-len 67 --d-model 256 --heads 4 --ffn-dim 1024 --layers 8 --warmup 10 --repeats 7 --benchmark-rounds 4"
```

## Limitations

* One of us didn't have access to an Nvidia GPU so we had to separate our development workflows then combine both solutions at the end
* Understanding of the transformer model and its complexity within the given time frame
* Sometimes Codex doesn't adhere to what was asked of it (e.g. asking for a CUDA implementation but it gives one that doesn't use CUDA)

## Potential Improvements with more time

* Improving both implementations for higher speedups
* Improving our understanding of Transformers to better understand the code
* Improving the technical report with benchmark values and diagrams
* Improving the documentation of our AI usage