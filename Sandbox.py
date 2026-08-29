#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import copy
import math
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional custom CUDA backend, adapted from cuda.py.
AUTO_CUDA_VALIDATED = True
AUTO_CUDA_BACKEND = "cuda-hybrid"
HYBRID_TF32_FFN_EXPANSION_LAYERS = 3
_cuda_extension = None


def get_cuda_extension(required: bool = True):
    """Load the pre-built CUDA extension without compiling it at runtime."""
    global _cuda_extension
    if _cuda_extension is not None:
        return _cuda_extension
    try:
        import transformer_cuda_ext
    except ImportError as error:
        if required:
            raise RuntimeError(
                "The CUDA backend is not built. Run `make build-cuda` first."
            ) from error
        return None
    _cuda_extension = transformer_cuda_ext
    return _cuda_extension


# Parse Cmd-Line Args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.01) # 1%
    parser.add_argument("--atol", type=float, default=0.001) # 0.1%
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark-rounds", type=int, default=2)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--optimized-backend",
        choices=("auto", "pytorch", "cuda", "cuda-hybrid"),
        default="auto",
        help="auto uses the custom CUDA backend only when its shape/build gate passes",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


# Resolve Device (CPU or Cuda)
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


# Resolve Data Type
def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


# Validate Cmd-Line Args, Device and Data Type
def validate_some(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


# Copy Model Weights (identical for BaselineTransformer and UserOptimizedTransformer)
def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None: # We will not mod this so all Transformer model param names (e.g. self.q_proj) must be kept the same
    state_dict = copy.deepcopy(baseline.state_dict()) # baseline.state_dict() gives state of BaselineTransformer mod and all its registered submods
    
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


# Try Compile Transformer Model
def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Gen Rand Test Case (used by run_accuracy_tests and benchmark_models)
def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


# Run Accuracy Tests
def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    @dataclass
    class AccuracyResult:
        passed: bool
        total_elements: int
        failed_elements: int
        max_abs_error: float
        max_relative_error: float
        mean_abs_error: float
        failed_feature_dims: List[int]
        worst_index: Tuple[int, ...]
        reference_at_worst: float
        optimized_at_worst: float


    def compare_outputs(
        reference: torch.Tensor,
        optimized: torch.Tensor,
        rtol: float,
        atol: float,
    ) -> AccuracyResult:
        if reference.shape != optimized.shape:
            raise AssertionError(
                f"shape mismatch: baseline={tuple(reference.shape)}, "
                f"optimized={tuple(optimized.shape)}"
            )
        if reference.dtype != optimized.dtype:
            print(
                f"[warning] dtype mismatch: baseline={reference.dtype}, "
                f"optimized={optimized.dtype}"
            )

        ref = reference.detach().float()
        opt = optimized.detach().float()

        finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
        abs_error = (opt - ref).abs()

        # Exact interpretation of the requested OR condition. torch.isclose uses
        # atol + rtol * abs(ref), which is slightly more permissive and is not used.
        abs_ok = abs_error <= atol
        rel_ok = abs_error <= rtol * ref.abs()
        passed_mask = finite_mask & (abs_ok | rel_ok)

        failed_mask = ~passed_mask
        failed_elements = int(failed_mask.sum().item())
        total_elements = reference.numel()

        flat_worst = int(abs_error.reshape(-1).argmax().item())
        worst_index_list = []
        remaining = flat_worst
        for size in reversed(reference.shape):
            worst_index_list.append(remaining % size)
            remaining //= size
        worst_index = tuple(reversed(worst_index_list))

        denominator = ref.abs().clamp_min(1e-12)
        relative_error = abs_error / denominator

        # Summarize failures by the last/output-feature dimension.
        if reference.ndim == 0:
            failed_feature_dims = [0] if failed_elements else []
        elif reference.ndim == 1:
            failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
        else:
            reduce_dims = tuple(range(reference.ndim - 1))
            failed_by_feature = failed_mask.any(dim=reduce_dims)
            failed_feature_dims = (
                torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
            )

        return AccuracyResult(
            passed=failed_elements == 0,
            total_elements=total_elements,
            failed_elements=failed_elements,
            max_abs_error=float(abs_error.max().item()),
            max_relative_error=float(relative_error.max().item()),
            mean_abs_error=float(abs_error.mean().item()),
            failed_feature_dims=failed_feature_dims,
            worst_index=worst_index,
            reference_at_worst=float(ref[worst_index].item()),
            optimized_at_worst=float(opt[worst_index].item()),
        )


    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


# Benchmark Models
def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    def warmup_model(
        model: nn.Module,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> None:
        with torch.inference_mode():
            for _ in range(iterations):
                model(x, valid_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


    def benchmark_once(
        model: nn.Module,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> List[float]:
        samples_ms: List[float] = []

        with torch.inference_mode():
            if device.type == "cuda":
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

                torch.cuda.synchronize(device)
                for index in range(iterations):
                    starts[index].record()
                    model(x, valid_mask)
                    ends[index].record()
                torch.cuda.synchronize(device)

                samples_ms.extend(
                    start.elapsed_time(end) for start, end in zip(starts, ends)
                )
            else:
                for _ in range(iterations):
                    start = time.perf_counter_ns()
                    model(x, valid_mask)
                    end = time.perf_counter_ns()
                    samples_ms.append((end - start) / 1e6)

        return samples_ms


    @dataclass
    class TimingResult:
        samples_ms: List[float]

        @property
        def mean_ms(self) -> float:
            return statistics.fmean(self.samples_ms)

        @property
        def median_ms(self) -> float:
            return statistics.median(self.samples_ms)

        @property
        def p90_ms(self) -> float:
            def _percentile(values: List[float], q: float) -> float:
                if not values:
                    raise ValueError("values must not be empty")
                ordered = sorted(values)
                if len(ordered) == 1:
                    return ordered[0]
                position = (len(ordered) - 1) * q
                lower = math.floor(position)
                upper = math.ceil(position)
                if lower == upper:
                    return ordered[lower]
                weight = position - lower
                return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


            return _percentile(self.samples_ms, 0.90)

        @property
        def min_ms(self) -> float:
            return min(self.samples_ms)


    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"Speedup (based on median latency): {speedup:.3f}x")


class BaselineTransformer(nn.Module): # nn.Module is base class
    class BaselineTransformerBlock(nn.Module):
        class BaselineSelfAttention(nn.Module):
            """Explicit multi-head self-attention implemented with native PyTorch ops."""

            def __init__(self, d_model: int, num_heads: int) -> None:
                super().__init__()
                if d_model % num_heads != 0:
                    raise ValueError("d_model must be divisible by num_heads")

                self.d_model = d_model
                self.num_heads = num_heads
                self.head_dim = d_model // num_heads
                self.scale = self.head_dim**-0.5

                self.q_proj = nn.Linear(d_model, d_model, bias=True)
                self.k_proj = nn.Linear(d_model, d_model, bias=True)
                self.v_proj = nn.Linear(d_model, d_model, bias=True)
                self.out_proj = nn.Linear(d_model, d_model, bias=True)

            def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
                batch, seq_len, _ = x.shape
                return (
                    x.view(batch, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )

            def forward(
                self,
                x: torch.Tensor,
                valid_token_mask: Optional[torch.Tensor] = None,
                causal: bool = False,
            ) -> torch.Tensor:
                batch, seq_len, _ = x.shape

                q = self._split_heads(self.q_proj(x))
                k = self._split_heads(self.k_proj(x))
                v = self._split_heads(self.v_proj(x))

                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                if causal:
                    causal_mask = torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1)
                    scores = scores.masked_fill(causal_mask, float("-inf"))

                if valid_token_mask is not None:
                    # Mask invalid key positions. Shape: [B, 1, 1, S].
                    invalid_keys = ~valid_token_mask[:, None, None, :]
                    scores = scores.masked_fill(invalid_keys, float("-inf"))

                # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
                probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
                context = torch.matmul(probs, v)
                context = (
                    context.transpose(1, 2)
                    .contiguous()
                    .view(batch, seq_len, self.d_model)
                )
                output = self.out_proj(context)

                if valid_token_mask is not None:
                    output = output.masked_fill(~valid_token_mask[..., None], 0)
                return output

        def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attention = self.BaselineSelfAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn_in = nn.Linear(d_model, ffn_dim)
            self.ffn_out = nn.Linear(ffn_dim, d_model)

        def forward(
            self,
            x: torch.Tensor,
            valid_token_mask: Optional[torch.Tensor],
            causal: bool,
        ) -> torch.Tensor:
            x = x + self.attention(self.norm1(x), valid_token_mask, causal)
            x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
            return x

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                self.BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x # Return tensor with shape [batch_size, seq_len, d_model]


class UserOptimizedTransformerBlock(nn.Module):
    """Standalone transformer block containing only optimized parameters."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)


class UserOptimizedTransformer(nn.Module):
    """Sandbox optimized Transformer with optional custom CUDA dispatch.

    backend="pytorch" always uses Sandbox's existing fused PyTorch forward.
    backend="cuda" or "cuda-hybrid" uses the custom CUDA path from cuda.py.
    backend="auto" selects the CUDA path only when its supported shape/backend
    is available; otherwise it uses Sandbox's existing PyTorch path.
    """

    def __init__(self, config: TransformerConfig, backend: str = "auto") -> None:
        super().__init__()
        if backend not in ("auto", "pytorch", "cuda", "cuda-hybrid"):
            raise ValueError(f"unknown backend: {backend}")
        self.config = config
        self.backend = backend
        self.layers = nn.ModuleList(
            UserOptimizedTransformerBlock(
                config.d_model, config.num_heads, config.ffn_dim
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self._d_model = config.d_model
        self._num_heads = config.num_heads

        self.register_buffer(
            "_qkv_weights",
            torch.empty(
                config.num_layers,
                3 * config.d_model,
                config.d_model,
                dtype=self.layers[0].q_proj.weight.dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_qkv_biases",
            torch.empty(
                config.num_layers,
                3 * config.d_model,
                dtype=self.layers[0].q_proj.bias.dtype,
            ),
            persistent=False,
        )

        self._causal_mask_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}
        self._mask_cache_key: Optional[Tuple[int, int]] = None
        self._mask_is_all_valid = False

        self.register_buffer(
            "_empty_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )

        self._pack_qkv_weights()

    @torch.no_grad()
    def _pack_qkv_weights(self) -> None:
        """Pack Sandbox's separate Q/K/V parameters for its fused PyTorch path."""
        d = self._d_model
        for i, layer in enumerate(self.layers):
            self._qkv_weights[i, :d].copy_(layer.q_proj.weight)
            self._qkv_weights[i, d:2 * d].copy_(layer.k_proj.weight)
            self._qkv_weights[i, 2 * d:].copy_(layer.v_proj.weight)
            self._qkv_biases[i, :d].copy_(layer.q_proj.bias)
            self._qkv_biases[i, d:2 * d].copy_(layer.k_proj.bias)
            self._qkv_biases[i, 2 * d:].copy_(layer.v_proj.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load baseline weights and rebuild packed QKV buffers."""
        remapped = state_dict.copy()
        for key in list(remapped):
            if key.startswith("layers.") and ".attention." in key:
                new_key = key.replace(".attention.", ".", 1)
                if new_key not in remapped:
                    remapped[new_key] = remapped[key]
                del remapped[key]

        result = super().load_state_dict(remapped, strict=strict, assign=assign)
        self._pack_qkv_weights()
        self._causal_mask_cache.clear()
        self._mask_cache_key = None
        return result

    def _valid_mask_or_none(
        self, valid_token_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if valid_token_mask is None:
            return None
        try:
            version = valid_token_mask._version
        except RuntimeError:
            version = -1
        key = (valid_token_mask.data_ptr(), version)
        if key != self._mask_cache_key:
            self._mask_cache_key = key
            self._mask_is_all_valid = bool(valid_token_mask.all().item())
        return None if self._mask_is_all_valid else valid_token_mask

    def _causal_mask(self, device: torch.device, seq_len: int) -> torch.Tensor:
        key = (device, seq_len)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.ones(
                (seq_len, seq_len), device=device, dtype=torch.bool
            ).triu(1)
            mask = mask.view(1, 1, seq_len, seq_len).expand(
                1, self._num_heads, seq_len, seq_len
            )
            self._causal_mask_cache[key] = mask
        return mask

    def _run_layer(
        self,
        layer_index: int,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_type: Optional[int],
    ) -> torch.Tensor:
        """Sandbox's existing fused PyTorch Transformer encoder primitive."""
        layer = self.layers[layer_index]
        return torch._transformer_encoder_layer_fwd(
            x,
            self._d_model,
            self._num_heads,
            self._qkv_weights[layer_index],
            self._qkv_biases[layer_index],
            layer.out_proj.weight,
            layer.out_proj.bias,
            True,
            True,
            layer.norm1.eps,
            layer.norm1.weight,
            layer.norm1.bias,
            layer.norm2.weight,
            layer.norm2.bias,
            layer.ffn_in.weight,
            layer.ffn_in.bias,
            layer.ffn_out.weight,
            layer.ffn_out.bias,
            mask,
            mask_type,
        )

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Original Sandbox forward implementation."""
        valid_token_mask = self._valid_mask_or_none(valid_token_mask)
        has_padding = valid_token_mask is not None

        if not self.config.causal and not has_padding:
            mask = None
            mask_type = None
        else:
            _, seq_len, _ = x.shape
            if self.config.causal:
                mask = self._causal_mask(x.device, seq_len)
                if has_padding:
                    mask = mask | (~valid_token_mask).view(1, 1, -1, seq_len)
                mask_type = 2
            else:
                mask = ~valid_token_mask
                mask_type = 1

        for i in range(len(self.layers)):
            x = self._run_layer(i, x, mask, mask_type)

        x = F.layer_norm(
            x,
            (self._d_model,),
            self.final_norm.weight,
            self.final_norm.bias,
            self.final_norm.eps,
        )
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CUDA inference implementation adapted from cuda.py."""
        if torch.is_grad_enabled():
            raise RuntimeError("The custom CUDA backend supports inference only")
        if not x.is_cuda or x.dtype != torch.float32:
            raise RuntimeError("The custom CUDA backend requires CUDA float32 input")

        extension = get_cuda_extension(required=True)
        mask = self._empty_mask
        attention_mask: Optional[torch.Tensor] = None
        if valid_token_mask is not None:
            mask = valid_token_mask.contiguous()
            attention_mask = mask[:, None, None, :]

        selective_tf32 = self.backend == "cuda-hybrid"
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if selective_tf32:
                torch.backends.cuda.matmul.allow_tf32 = False

            normalized = self.layers[0].norm1(x)
            for index, layer in enumerate(self.layers):
                if selective_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = False

                batch, seq_len, _ = normalized.shape
                head_dim = self._d_model // self._num_heads
                qkv = F.linear(
                    normalized,
                    self._qkv_weights[index],
                    self._qkv_biases[index],
                ).view(batch, seq_len, 3, self._num_heads, head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

                # Match cuda.py's attention computation. When padding is present
                # together with causal attention, combine both restrictions into
                # one explicit boolean mask because SDPA does not accept both a
                # non-None attn_mask and is_causal=True simultaneously.
                if self.config.causal and attention_mask is not None:
                    causal = torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).triu(1)
                    allowed = (~causal)[None, None, :, :] & attention_mask
                    context = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False
                    )
                else:
                    context = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=self.config.causal,
                    )
                context = context.transpose(1, 2).reshape(
                    batch, seq_len, self._d_model
                )
                attention_delta = F.linear(
                    context, layer.out_proj.weight, bias=None
                )

                x, ffn_input = extension.residual_bias_layer_norm(
                    x,
                    attention_delta,
                    layer.out_proj.bias,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    mask,
                    layer.norm2.eps,
                    False,
                )

                if selective_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = (
                        index < HYBRID_TF32_FFN_EXPANSION_LAYERS
                    )
                hidden = F.linear(ffn_input, layer.ffn_in.weight, bias=None)
                hidden = extension.bias_gelu(hidden, layer.ffn_in.bias)
                if selective_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                ffn_delta = F.linear(hidden, layer.ffn_out.weight, bias=None)

                last_layer = index + 1 == len(self.layers)
                following_norm = (
                    self.final_norm if last_layer else self.layers[index + 1].norm1
                )
                x, normalized = extension.residual_bias_layer_norm(
                    x,
                    ffn_delta,
                    layer.ffn_out.bias,
                    following_norm.weight,
                    following_norm.bias,
                    mask,
                    following_norm.eps,
                    last_layer,
                )

            return normalized
        finally:
            if selective_tf32:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Explicit backend selection takes precedence. Auto uses CUDA only when
        # the CUDA implementation is actually eligible and the extension exists.
        if self.backend == "pytorch":
            return self._forward_pytorch(x, valid_token_mask)
        if self.backend in ("cuda", "cuda-hybrid"):
            return self._forward_cuda(x, valid_token_mask)

        # backend == "auto"
        if x.is_cuda and x.dtype == torch.float32:
            supported = custom_cuda_shape_supported(self.config, x.device, x.dtype)
            if supported and AUTO_CUDA_VALIDATED and get_cuda_extension(required=False):
                original_backend = self.backend
                self.backend = AUTO_CUDA_BACKEND
                try:
                    return self._forward_cuda(x, valid_token_mask)
                finally:
                    self.backend = original_backend

        return self._forward_pytorch(x, valid_token_mask)


def custom_cuda_shape_supported(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    """Shape/device gate copied from cuda.py for the custom kernels."""
    return (
        device.type == "cuda"
        and dtype == torch.float32
        and config.batch_size == 8
        and config.seq_len == 128
        and config.d_model == 512
        and config.num_heads == 8
        and config.ffn_dim == 2048
        and config.num_layers == 6
    )


def resolve_optimized_backend(
    requested: str,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    if requested == "pytorch":
        return requested

    supported = custom_cuda_shape_supported(config, device, dtype)
    if requested in ("cuda", "cuda-hybrid"):
        if not supported:
            raise ValueError(
                f"--optimized-backend {requested} requires CUDA float32 and the default "
                "8x128x512, 8-head, 2048-FFN, 6-layer configuration"
            )
        get_cuda_extension(required=True)
        return requested

    if supported and AUTO_CUDA_VALIDATED and get_cuda_extension(required=False):
        return AUTO_CUDA_BACKEND
    return "pytorch"


# Entry Pt
def main() -> int:
    # Start: Initial setup
    args = parse_args()
    
    device = resolve_device(args.device)
    
    dtype = resolve_dtype(args.dtype)

    validate_some(args, device, dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )

    config.validate()
    # End: Initial setup

    # Start: PyTorch setup
    torch.manual_seed(args.seed)
    
    torch.set_float32_matmul_precision(args.matmul_precision)
    
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    # End: PyTorch setup

    optimized_backend = resolve_optimized_backend(
        args.optimized_backend, config, device, dtype
    )

    # Start: Logging of setup info
    print("=== Configuration ===")
    
    print(config)
    
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    
    print(f"optimized_backend={optimized_backend} (requested={args.optimized_backend})")

    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    # End: Logging of setup info

    baseline = BaselineTransformer(config)
    
    optimized = UserOptimizedTransformer(config, backend=optimized_backend)
    
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    
    optimized = optimized.to(device=device, dtype=dtype).eval()

    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    
    compile_user = args.compile_user
    if optimized_backend in ("cuda", "cuda-hybrid") and compile_user:
        print(
            "[warning] --compile-user is ignored for the custom CUDA backend; "
            "its CUDA path is already specialized for inference"
        )
        compile_user = False
    optimized = maybe_compile(optimized, compile_user, args.compile_mode)

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol, # Correct if abs(user - ref) <= rtol * abs(ref)
        atol=args.atol, # Correct if abs(user - ref) <= atol
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )

    return 0 if accuracy_passed else 2


# Ensures this script is ran directly and not imported as a mod
if __name__ == "__main__":
    raise SystemExit(main())
