# Transformer GPU Optimization Challenge

Transformers power many modern AI systems across language, vision, speech, recommendation, and large language models. Their key operation is self-attention, which lets every token interact directly with every other token while enabling parallel GPU processing.

For an input $X \in \mathbb{R}^{N \times d}$, the model projects the input into query, key, and value matrices:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Scaled dot-product attention is then computed as:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The scaling term $\sqrt{d_k}$ keeps dot products from becoming too large and destabilizing softmax.

The competition asks participants to use AI-assisted techniques to optimize Transformer runtime on a specified GPU while keeping results numerically correct relative to the reference implementation. Key costs include matrix multiplication, attention-score calculation, softmax, normalization, and feed-forward layers. Performance may be constrained by compute throughput, memory bandwidth, cache efficiency, kernel-launch overhead, and tensor-core utilization.

Possible optimization approaches include operator fusion, improved memory layouts, reduced-precision computation, tensor-core usage, optimized softmax, and custom CUDA, Triton, TensorFlow, or PyTorch implementations. The broader goal is to demonstrate how AI can help analyze Transformer workloads, locate bottlenecks, and produce hardware-specific performance improvements.

`tensorflow_transformer_benchmark.py` will not be used at all, focus work on `torch_transformer_benchmark.py`.