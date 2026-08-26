## Setup
* python3 -m venv venv
* source venv/bin/activate
* pip install numpy
* pip install torch

## Run
* python3 Sandbox.py --warmup 7 --repeats 5 --benchmark-rounds 2

## Possible Optimizations
* [Overall] Custom CUDA, Triton, TensorFlow or PyTorch implementations
* torch.nn.functional.scaled_dot_product_attention
* torch.compile
* Triton/CUDA fused kernels
* fused LayerNorm / residual / FFN
* operator fusion
* memory layout optimization
* reduced-precision computation
* tensor core usage
* softmax optimization