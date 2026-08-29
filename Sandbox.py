#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# Set only after the CUDA backend beats the compiled PyTorch implementation by
# at least 3% on the target GPU and passes the complete accuracy matrix.
AUTO_CUDA_VALIDATED = True
AUTO_CUDA_BACKEND = "cuda-hybrid"
AGGRESSIVE_CUDA_VALIDATED = False
AGGRESSIVE_LT_FFN_VALIDATED = False
_cuda_extension = None


@dataclass(frozen=True)
class GemmPrecisionPlan:
    """Per-layer TF32 policy for the four explicit GEMMs in each block."""

    qkv: Tuple[bool, ...]
    attention_out: Tuple[bool, ...]
    ffn_in: Tuple[bool, ...]
    ffn_out: Tuple[bool, ...]

    def validate(self, layers: int) -> None:
        for name, values in (
            ("qkv", self.qkv),
            ("attention_out", self.attention_out),
            ("ffn_in", self.ffn_in),
            ("ffn_out", self.ffn_out),
        ):
            if len(values) != layers:
                raise ValueError(f"precision plan {name} must contain {layers} values")

    def to_mask(self) -> int:
        mask = 0
        offset = 0
        for values in (self.qkv, self.attention_out, self.ffn_in, self.ffn_out):
            for enabled in values:
                if enabled:
                    mask |= 1 << offset
                offset += 1
        return mask

    @classmethod
    def from_mask(cls, mask: int, layers: int) -> "GemmPrecisionPlan":
        if mask < 0 or mask >= (1 << (4 * layers)):
            raise ValueError(f"precision mask must fit in {4 * layers} bits")
        groups = []
        offset = 0
        for _ in range(4):
            groups.append(tuple(bool(mask & (1 << (offset + i))) for i in range(layers)))
            offset += layers
        return cls(*groups)

    @classmethod
    def strict(cls, layers: int) -> "GemmPrecisionPlan":
        disabled = (False,) * layers
        return cls(disabled, disabled, disabled, disabled)

    @classmethod
    def validated_hybrid(cls, layers: int) -> "GemmPrecisionPlan":
        # This reproduces the previously validated policy: strict attention,
        # TF32 for the first three FFN expansions, and every FFN contraction.
        disabled = (False,) * layers
        return cls(
            disabled,
            disabled,
            tuple(index < 3 for index in range(layers)),
            (True,) * layers,
        )

    def describe(self) -> str:
        def enabled(values: Tuple[bool, ...]) -> str:
            indices = [str(i) for i, value in enumerate(values) if value]
            return ",".join(indices) if indices else "none"

        return (
            f"mask=0x{self.to_mask():x} qkv=[{enabled(self.qkv)}] "
            f"attention_out=[{enabled(self.attention_out)}] "
            f"ffn_in=[{enabled(self.ffn_in)}] ffn_out=[{enabled(self.ffn_out)}]"
        )


# Updated only after `make tune-aggressive` passes the complete target-GPU gate.
VALIDATED_AGGRESSIVE_TF32_MASK = GemmPrecisionPlan.validated_hybrid(6).to_mask()


def get_cuda_extension(required: bool = True):
    """Load the ahead-of-time CUDA extension without compiling at runtime."""
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


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
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


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
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
        return x


class OptimizedSelfAttention(nn.Module):
    """Self-attention using one QKV projection and PyTorch's fused SDPA."""

    def __init__(
        self, d_model: int, num_heads: int, force_flash: bool = False
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.force_flash = force_flash
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        include_output_bias: bool = True,
        qkv_tf32: Optional[bool] = None,
        output_tf32: Optional[bool] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        if qkv_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = qkv_tf32
        # [B, S, 3D] -> three [B, H, S, Dh] views. Keeping the head dimension
        # before sequence is the layout expected by scaled_dot_product_attention.
        qkv = self.qkv_proj(x).view(
            batch, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if qkv_tf32 is not None:
            # Keep the attention score/value kernels conservative; the qkv bit
            # controls only the projection GEMM and is independently searchable.
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.force_flash:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                )
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=causal,
            )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        if output_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = output_tf32
        output = F.linear(
            context,
            self.out_proj.weight,
            self.out_proj.bias if include_output_bias else None,
        )

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class OptimizedTransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, force_flash: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = OptimizedSelfAttention(d_model, num_heads, force_flash)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x), attention_mask, valid_token_mask, causal
        )
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(nn.Module):
    """Transformer with fused QKV/SDPA and an optional custom CUDA backend."""

    def __init__(
        self,
        config: TransformerConfig,
        backend: str = "pytorch",
        precision_plan: Optional[GemmPrecisionPlan] = None,
        experimental_lt_ffn: bool = False,
        enable_cuda_graph: bool = True,
        force_flash: bool = False,
        profile_ranges: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        if precision_plan is None:
            if backend == "cuda":
                precision_plan = GemmPrecisionPlan.strict(config.num_layers)
            elif backend == "cuda-hybrid":
                precision_plan = GemmPrecisionPlan.validated_hybrid(config.num_layers)
            elif backend == "cuda-aggressive":
                precision_plan = GemmPrecisionPlan.from_mask(
                    VALIDATED_AGGRESSIVE_TF32_MASK, config.num_layers
                )
            else:
                precision_plan = GemmPrecisionPlan.strict(config.num_layers)
        precision_plan.validate(config.num_layers)
        self.precision_plan = precision_plan
        self.experimental_lt_ffn = experimental_lt_ffn
        self.enable_cuda_graph = enable_cuda_graph
        self.profile_ranges = profile_ranges
        self._cuda_graph_states: Dict[bool, Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.layers = nn.ModuleList(
            [
                OptimizedTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim, force_flash
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.register_buffer(
            "_empty_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "_lt_workspace", torch.empty(0, dtype=torch.uint8), persistent=False
        )

    def _apply(self, fn, recurse: bool = True):
        # Captured graph tensors are deliberately not module buffers. Discard
        # them before a device/dtype move so stale device pointers cannot replay.
        self._cuda_graph_states.clear()
        return super()._apply(fn, recurse=recurse)

    def set_precision_plan(self, plan: GemmPrecisionPlan) -> None:
        """Install an offline-search candidate and invalidate captured graphs."""
        plan.validate(self.config.num_layers)
        self.precision_plan = plan
        self._cuda_graph_states.clear()

    def prepare_aggressive_backend(self) -> None:
        """Allocate and initialize optional resources before graph capture."""
        if self.backend != "cuda-aggressive" or not self.experimental_lt_ffn:
            return
        if self._empty_mask.device.type != "cuda":
            raise RuntimeError("move the aggressive model to CUDA before preparing it")
        if self._lt_workspace.numel() == 0:
            # cuBLASLt's heuristic is allowed up to 32 MiB and is selected only
            # once. The allocation is persistent and excluded from inference.
            self._lt_workspace = torch.empty(
                32 * 1024 * 1024,
                dtype=torch.uint8,
                device=self._empty_mask.device,
            )
        extension = get_cuda_extension(required=True)
        if not hasattr(extension, "ffn_gelu_lt"):
            raise RuntimeError("rebuild the CUDA extension to enable cuBLASLt FFN")
        # Initialize both cached compute policies before CUDA Graph capture.
        sample = torch.zeros(
            self.config.batch_size,
            self.config.seq_len,
            self.config.d_model,
            dtype=torch.float32,
            device=self._empty_mask.device,
        )
        first = self.layers[0].ffn_in
        extension.ffn_gelu_lt(
            sample, first.weight, first.bias, self._lt_workspace, False
        )
        extension.ffn_gelu_lt(
            sample, first.weight, first.bias, self._lt_workspace, True
        )
        torch.cuda.synchronize(sample.device)

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_mask: Optional[torch.Tensor] = None

        if valid_token_mask is not None:
            # SDPA boolean masks use True for elements that are allowed to
            # participate. This masks keys; query rows are zeroed below.
            attention_mask = valid_token_mask[:, None, None, :]

        for layer in self.layers:
            x = layer(
                x,
                attention_mask,
                valid_token_mask,
                self.config.causal,
            )
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _forward_cuda_impl(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("The custom CUDA backend supports inference only")
        extension = get_cuda_extension(required=True)
        mask = self._empty_mask
        attention_mask: Optional[torch.Tensor] = None
        if valid_token_mask is not None:
            mask = valid_token_mask.contiguous()
            attention_mask = mask[:, None, None, :]

        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False

            # The first LayerNorm cannot be fused with a preceding block. Each
            # subsequent norm is produced by the previous residual fusion.
            normalized = self.layers[0].norm1(x)
            for index, layer in enumerate(self.layers):
                if self.profile_ranges:
                    torch.cuda.nvtx.range_push(f"transformer_layer_{index}")
                    torch.cuda.nvtx.range_push("attention")
                attention_delta = layer.attention(
                    normalized,
                    attention_mask,
                    valid_token_mask,
                    self.config.causal,
                    include_output_bias=False,
                    qkv_tf32=self.precision_plan.qkv[index],
                    output_tf32=self.precision_plan.attention_out[index],
                )
                if self.profile_ranges:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("attention_residual_layernorm")
                x, ffn_input = extension.residual_bias_layer_norm(
                    x,
                    attention_delta,
                    layer.attention.out_proj.bias,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    mask,
                    layer.norm2.eps,
                    False,
                )
                if self.profile_ranges:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("ffn")

                torch.backends.cuda.matmul.allow_tf32 = self.precision_plan.ffn_in[index]
                if self.experimental_lt_ffn:
                    hidden = extension.ffn_gelu_lt(
                        ffn_input,
                        layer.ffn_in.weight,
                        layer.ffn_in.bias,
                        self._lt_workspace,
                        self.precision_plan.ffn_in[index],
                    )
                else:
                    hidden = F.linear(ffn_input, layer.ffn_in.weight, bias=None)
                    hidden = extension.bias_gelu(hidden, layer.ffn_in.bias)
                torch.backends.cuda.matmul.allow_tf32 = self.precision_plan.ffn_out[index]
                ffn_delta = F.linear(hidden, layer.ffn_out.weight, bias=None)
                if self.profile_ranges:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("ffn_residual_layernorm")

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
                if self.profile_ranges:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
            return normalized
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def _capture_cuda_graph(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture one mask topology with persistent input/output storage."""
        static_x = torch.empty_like(x)
        if valid_token_mask is None:
            static_mask = self._empty_mask
            capture_mask = None
        else:
            static_mask = torch.empty_like(valid_token_mask, memory_format=torch.contiguous_format)
            capture_mask = static_mask
        static_x.copy_(x)
        if valid_token_mask is not None:
            static_mask.copy_(valid_token_mask)

        # Lazy CUDA libraries and cuBLAS workspaces must be initialized on a
        # side stream before capture. This happens once per mask topology.
        capture_stream = torch.cuda.Stream(device=x.device)
        capture_stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._forward_cuda_impl(static_x, capture_mask)
        torch.cuda.current_stream(x.device).wait_stream(capture_stream)
        torch.cuda.synchronize(x.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self._forward_cuda_impl(static_x, capture_mask)
        return graph, static_x, static_mask, static_output

    def _forward_cuda_graph(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        key = valid_token_mask is not None
        state = self._cuda_graph_states.get(key)
        if state is None:
            state = self._capture_cuda_graph(x, valid_token_mask)
            self._cuda_graph_states[key] = state
        graph, static_x, static_mask, static_output = state
        static_x.copy_(x)
        if valid_token_mask is not None:
            static_mask.copy_(valid_token_mask)
        graph.replay()
        return static_output

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.backend == "cuda-aggressive" and self.enable_cuda_graph:
            return self._forward_cuda_graph(x, valid_token_mask)
        return self._forward_cuda_impl(x, valid_token_mask)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.backend in ("cuda", "cuda-hybrid", "cuda-aggressive"):
            if not x.is_cuda or x.dtype != torch.float32:
                raise RuntimeError("The custom backend requires CUDA float32 input")
            return self._forward_cuda(x, valid_token_mask)
        return self._forward_pytorch(x, valid_token_mask)


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    if isinstance(baseline, BaselineTransformer) and isinstance(
        optimized, UserOptimizedTransformer
    ):
        if len(baseline.layers) != len(optimized.layers):
            raise ValueError("baseline and optimized layer counts differ")
        with torch.no_grad():
            optimized.final_norm.load_state_dict(baseline.final_norm.state_dict())
            for source, target in zip(baseline.layers, optimized.layers):
                target.norm1.load_state_dict(source.norm1.state_dict())
                target.norm2.load_state_dict(source.norm2.state_dict())
                target.ffn_in.load_state_dict(source.ffn_in.state_dict())
                target.ffn_out.load_state_dict(source.ffn_out.state_dict())
                target.attention.out_proj.load_state_dict(
                    source.attention.out_proj.state_dict()
                )
                target.attention.qkv_proj.weight.copy_(
                    torch.cat(
                        (
                            source.attention.q_proj.weight,
                            source.attention.k_proj.weight,
                            source.attention.v_proj.weight,
                        ),
                        dim=0,
                    )
                )
                target.attention.qkv_proj.bias.copy_(
                    torch.cat(
                        (
                            source.attention.q_proj.bias,
                            source.attention.k_proj.bias,
                            source.attention.v_proj.bias,
                        ),
                        dim=0,
                    )
                )
        return

    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def custom_cuda_shape_supported(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    """Whether the shape-specialized RTX 5090 kernels can execute this case."""
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
    if requested in ("cuda", "cuda-hybrid", "cuda-aggressive"):
        if not supported:
            raise ValueError(
                f"--optimized-backend {requested} requires CUDA float32 and the default "
                "8x128x512, 8-head, 2048-FFN, 6-layer configuration"
            )
        get_cuda_extension(required=True)
        return requested

    # Auto remains conservative: an extension must be present, eligible, and
    # have cleared the documented accuracy/performance gate on the target GPU.
    if supported and get_cuda_extension(required=False):
        if AGGRESSIVE_CUDA_VALIDATED:
            return "cuda-aggressive"
        if AUTO_CUDA_VALIDATED:
            return AUTO_CUDA_BACKEND
    return "pytorch"


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        # None avoids passing an all-True mask through every layer and lets
        # causal SDPA use its fastest specialized path.
        return x, None

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


def run_cuda_graph_replay_tests(
    baseline: nn.Module,
    optimized: UserOptimizedTransformer,
    config: TransformerConfig,
    device: torch.device,
    rtol: float,
    atol: float,
    seed: int,
) -> bool:
    """Exercise graph input copying and both optional-mask topologies."""
    if optimized.backend != "cuda-aggressive" or not optimized.enable_cuda_graph:
        return True
    print("\n=== CUDA Graph replay checks ===")
    checks: List[Tuple[str, torch.Tensor, Optional[torch.Tensor]]] = []
    x, _ = generate_random_case(
        config, device, torch.float32, seed + 70000, 0.0, 1.0
    )
    checks.append(("initial allocation", x, None))
    checks.append(("different allocation", x.clone().mul_(0.75), None))
    masked_x, mask = generate_random_case(
        config, device, torch.float32, seed + 70001, 0.35, 1.0
    )
    checks.append(("masked allocation", masked_x, mask))
    changed_mask = mask.clone() if mask is not None else None
    if changed_mask is not None:
        changed_mask[:, -8:] = False
    checks.append(("changed mask values", masked_x, changed_mask))

    passed = True
    with torch.inference_mode():
        for label, case_x, case_mask in checks:
            reference = baseline(case_x, case_mask)
            candidate = optimized(case_x, case_mask).clone()
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)
            passed &= result.passed
            print(
                f"{label}: {'PASS' if result.passed else 'FAIL'} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

        # Mutate the same allocation after its graph topology has been cached.
        x.add_(0.03125)
        result = compare_outputs(
            baseline(x, None), optimized(x, None).clone(), rtol=rtol, atol=atol
        )
        passed &= result.passed
        print(
            f"same allocation mutated: {'PASS' if result.passed else 'FAIL'} | "
            f"max_abs={result.max_abs_error:.6g} | "
            f"failed={result.failed_elements}/{result.total_elements}"
        )
    return passed


def percentile(values: List[float], q: float) -> float:
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
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
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
    valid_mask: Optional[torch.Tensor],
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


def measure_peak_activation_memory(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    device: torch.device,
) -> int:
    """Return peak CUDA bytes allocated above the steady-state allocation."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    steady_state = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(x, valid_mask)
    torch.cuda.synchronize(device)
    return max(0, torch.cuda.max_memory_allocated(device) - steady_state)


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

    baseline_peak_bytes = measure_peak_activation_memory(
        baseline, x, valid_mask, device
    )
    optimized_peak_bytes = measure_peak_activation_memory(
        optimized, x, valid_mask, device
    )

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
    print(f"speedup  : {speedup:.3f}x based on median latency")
    if device.type == "cuda":
        mib = 1024.0 * 1024.0
        print(
            f"peak activation allocation: baseline={baseline_peak_bytes / mib:.2f} MiB | "
            f"optimized={optimized_peak_bytes / mib:.2f} MiB"
        )


@dataclass
class MatrixAccuracyResult:
    passed: bool
    cases: int
    failed_elements: int
    total_elements: int
    max_abs_error: float
    max_relative_error: float


class ExpandedAccuracyHarness:
    """Cache the 2*2*3*trials stress matrix for offline CUDA tuning."""

    def __init__(
        self,
        config: TransformerConfig,
        device: torch.device,
        trials: int,
        seed: int,
        rtol: float,
        atol: float,
        use_lt_ffn: bool,
    ) -> None:
        self.device = device
        self.rtol = rtol
        self.atol = atol
        self.entries: List[
            Tuple[UserOptimizedTransformer, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
        ] = []
        self.timing_entry: Optional[
            Tuple[UserOptimizedTransformer, torch.Tensor, Optional[torch.Tensor]]
        ] = None

        for causal in (False, True):
            case_config = replace(config, causal=causal)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            baseline = BaselineTransformer(case_config).to(device=device).eval()
            optimized = UserOptimizedTransformer(
                case_config,
                backend="cuda-aggressive",
                precision_plan=GemmPrecisionPlan.validated_hybrid(
                    case_config.num_layers
                ),
                experimental_lt_ffn=use_lt_ffn,
                enable_cuda_graph=False,
            )
            copy_model_weights(baseline, optimized)
            optimized = optimized.to(device=device).eval()
            optimized.prepare_aggressive_backend()

            with torch.inference_mode():
                for padding_ratio in (0.0, 0.35):
                    for input_scale in (0.25, 1.0, 4.0):
                        for trial in range(trials):
                            x, mask = generate_random_case(
                                case_config,
                                device,
                                torch.float32,
                                seed + trial,
                                padding_ratio,
                                input_scale,
                            )
                            reference = baseline(x, mask)
                            self.entries.append((optimized, x, mask, reference))
                            if (
                                self.timing_entry is None
                                and not causal
                                and padding_ratio == 0.0
                                and input_scale == 1.0
                            ):
                                self.timing_entry = (optimized, x, mask)

    def evaluate(
        self, plan: GemmPrecisionPlan, verbose: bool = False
    ) -> MatrixAccuracyResult:
        models = {id(entry[0]): entry[0] for entry in self.entries}.values()
        for model in models:
            model.set_precision_plan(plan)

        failed_elements = 0
        total_elements = 0
        max_abs_error = 0.0
        max_relative_error = 0.0
        with torch.inference_mode():
            for index, (model, x, mask, reference) in enumerate(self.entries):
                result = compare_outputs(
                    reference, model(x, mask), rtol=self.rtol, atol=self.atol
                )
                failed_elements += result.failed_elements
                total_elements += result.total_elements
                max_abs_error = max(max_abs_error, result.max_abs_error)
                max_relative_error = max(
                    max_relative_error, result.max_relative_error
                )
                if verbose:
                    print(
                        f"matrix case {index + 1:02d}/{len(self.entries)}: "
                        f"{'PASS' if result.passed else 'FAIL'} | "
                        f"max_abs={result.max_abs_error:.6g} | "
                        f"failed={result.failed_elements}/{result.total_elements}"
                    )
        return MatrixAccuracyResult(
            passed=failed_elements == 0,
            cases=len(self.entries),
            failed_elements=failed_elements,
            total_elements=total_elements,
            max_abs_error=max_abs_error,
            max_relative_error=max_relative_error,
        )

    def time_plan(
        self, plan: GemmPrecisionPlan, warmup: int = 10, repeats: int = 30
    ) -> TimingResult:
        if self.timing_entry is None:
            raise RuntimeError("expanded matrix has no timing case")
        model, x, mask = self.timing_entry
        model.set_precision_plan(plan)
        warmup_model(model, x, mask, warmup, self.device)
        return TimingResult(benchmark_once(model, x, mask, repeats, self.device))


def search_tf32_precision_plan(
    config: TransformerConfig,
    device: torch.device,
    trials: int,
    seed: int,
    rtol: float,
    atol: float,
    beam_width: int,
    steps: int,
    tuning_output: Optional[Path],
) -> bool:
    """Bounded beam search; never executes in the normal inference path."""
    print("\n=== Offline aggressive TF32 search ===")
    print(
        "building the expanded reference matrix "
        f"(2 causal * 2 padding * 3 scales * {trials} seeds)"
    )
    exact_harness = ExpandedAccuracyHarness(
        config, device, trials, seed, rtol, atol, use_lt_ffn=False
    )
    initial = GemmPrecisionPlan.validated_hybrid(config.num_layers)
    initial_accuracy = exact_harness.evaluate(initial)
    if not initial_accuracy.passed:
        print(
            "validated hybrid seed failed expanded matrix: "
            f"failed={initial_accuracy.failed_elements}/{initial_accuracy.total_elements}"
        )
        return False
    initial_timing = exact_harness.time_plan(initial)
    cache: Dict[int, Tuple[MatrixAccuracyResult, TimingResult]] = {
        initial.to_mask(): (initial_accuracy, initial_timing)
    }
    beam: List[GemmPrecisionPlan] = [initial]
    best = initial

    for step in range(steps):
        proposals: Dict[int, GemmPrecisionPlan] = {}
        for plan in beam:
            mask = plan.to_mask()
            for bit in range(4 * config.num_layers):
                if not mask & (1 << bit):
                    candidate_mask = mask | (1 << bit)
                    proposals[candidate_mask] = GemmPrecisionPlan.from_mask(
                        candidate_mask, config.num_layers
                    )
        passing: List[Tuple[float, GemmPrecisionPlan]] = []
        for index, (mask, candidate) in enumerate(proposals.items()):
            if mask not in cache:
                accuracy = exact_harness.evaluate(candidate)
                if accuracy.passed:
                    timing = exact_harness.time_plan(candidate)
                    cache[mask] = (accuracy, timing)
                else:
                    cache[mask] = (accuracy, TimingResult([float("inf")]))
            accuracy, timing = cache[mask]
            print(
                f"step={step + 1} candidate={index + 1}/{len(proposals)} "
                f"mask=0x{mask:x} {'PASS' if accuracy.passed else 'FAIL'} "
                f"median={timing.median_ms:.4f} ms"
            )
            if accuracy.passed:
                passing.append((timing.median_ms, candidate))
        if not passing:
            break
        passing.sort(key=lambda item: item[0])
        beam = [candidate for _, candidate in passing[:beam_width]]
        if passing[0][0] < cache[best.to_mask()][1].median_ms:
            best = passing[0][1]

    best_accuracy, best_timing = cache[best.to_mask()]
    print(f"winning precision plan: {best.describe()}")
    print(
        f"exact-GELU median={best_timing.median_ms:.4f} ms, "
        f"p90={best_timing.p90_ms:.4f} ms, max_abs={best_accuracy.max_abs_error:.6g}"
    )

    # Independently gate the cuBLASLt GELU epilogue against the winning exact
    # path. It is recommended only with full correctness, >=3% median gain, and
    # no more than 2% p90 regression.
    lt_harness = ExpandedAccuracyHarness(
        config, device, trials, seed, rtol, atol, use_lt_ffn=True
    )
    lt_accuracy = lt_harness.evaluate(best)
    lt_timing = lt_harness.time_plan(best) if lt_accuracy.passed else None
    lt_promoted = bool(
        lt_timing is not None
        and lt_timing.median_ms <= best_timing.median_ms * 0.97
        and lt_timing.p90_ms <= best_timing.p90_ms * 1.02
    )
    print(
        "cuBLASLt FFN gate: "
        f"{'PROMOTE' if lt_promoted else 'KEEP EXACT GELU'} | "
        f"accuracy={'PASS' if lt_accuracy.passed else 'FAIL'} | "
        f"median={lt_timing.median_ms if lt_timing else float('nan'):.4f} ms"
    )

    report = {
        "precision_mask": best.to_mask(),
        "precision_mask_hex": hex(best.to_mask()),
        "precision_plan": best.describe(),
        "accuracy": best_accuracy.__dict__,
        "exact_timing_ms": {
            "median": best_timing.median_ms,
            "p90": best_timing.p90_ms,
            "minimum": best_timing.min_ms,
        },
        "cublaslt_ffn_promoted": lt_promoted,
        "cublaslt_accuracy": lt_accuracy.__dict__,
        "cublaslt_timing_ms": None
        if lt_timing is None
        else {
            "median": lt_timing.median_ms,
            "p90": lt_timing.p90_ms,
            "minimum": lt_timing.min_ms,
        },
    }
    if tuning_output is not None:
        tuning_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote tuning report to {tuning_output}")
    return True


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


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
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--optimized-backend",
        choices=("auto", "pytorch", "cuda", "cuda-hybrid", "cuda-aggressive"),
        default="auto",
        help="auto uses only a CUDA backend that has passed the speed gate",
    )
    parser.add_argument(
        "--precision-mask",
        type=lambda value: int(value, 0),
        help="override the custom backend's 4*layers TF32 bit mask (offline testing)",
    )
    parser.add_argument(
        "--experimental-lt-ffn",
        action="store_true",
        help="test the unpromoted cuBLASLt fused FFN expansion",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="disable CUDA Graph replay for cuda-aggressive",
    )
    parser.add_argument(
        "--force-flash-attention",
        action="store_true",
        help="require the fused Flash SDPA backend; fail if this FP32 case is unsupported",
    )
    parser.add_argument(
        "--profile-ranges",
        action="store_true",
        help="add per-layer attention/FFN NVTX ranges (use with CUDA Graph disabled)",
    )
    parser.add_argument(
        "--expanded-accuracy",
        action="store_true",
        help="run causal/non-causal, padding, and input-scale stress validation",
    )
    parser.add_argument(
        "--search-tf32-plan",
        action="store_true",
        help="offline beam search for a faster accuracy-safe per-GEMM TF32 plan",
    )
    parser.add_argument("--search-beam-width", type=int, default=4)
    parser.add_argument("--search-steps", type=int, default=4)
    parser.add_argument(
        "--tuning-output",
        type=Path,
        help="write offline aggressive-tuning results as JSON",
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
        default=False,
        help=(
            "enable/disable TF32 on CUDA for both implementations; disabled by "
            "default because TF32 can exceed the strict accuracy tolerance"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
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
    if args.search_beam_width <= 0 or args.search_steps <= 0:
        raise ValueError("search beam width and steps must be positive")
    if args.experimental_lt_ffn and device.type != "cuda":
        raise ValueError("--experimental-lt-ffn requires CUDA")
    if args.experimental_lt_ffn and args.optimized_backend not in (
        "cuda-aggressive",
        "auto",
    ):
        raise ValueError("--experimental-lt-ffn is only valid for cuda-aggressive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

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
    validate_args(args, device, dtype)
    optimized_backend = resolve_optimized_backend(
        args.optimized_backend, config, device, dtype
    )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if args.search_tf32_plan:
        if not custom_cuda_shape_supported(config, device, dtype):
            raise ValueError("TF32 search requires the default CUDA float32 shape")
        get_cuda_extension(required=True)
        return 0 if search_tf32_precision_plan(
            config=config,
            device=device,
            trials=args.accuracy_trials,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
            beam_width=args.search_beam_width,
            steps=args.search_steps,
            tuning_output=args.tuning_output,
        ) else 2

    precision_plan = None
    if args.precision_mask is not None:
        precision_plan = GemmPrecisionPlan.from_mask(
            args.precision_mask, config.num_layers
        )
    use_lt_ffn = args.experimental_lt_ffn or (
        optimized_backend == "cuda-aggressive" and AGGRESSIVE_LT_FFN_VALIDATED
    )
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(
        config,
        backend=optimized_backend,
        precision_plan=precision_plan,
        experimental_lt_ffn=use_lt_ffn,
        enable_cuda_graph=not args.no_cuda_graph,
        force_flash=args.force_flash_attention,
        profile_ranges=args.profile_ranges,
    )
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    if optimized_backend == "cuda-aggressive":
        optimized.prepare_aggressive_backend()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    compile_user = args.compile_user
    if optimized_backend in ("cuda", "cuda-hybrid", "cuda-aggressive") and compile_user:
        print(
            "[warning] --compile-user is ignored for the custom CUDA backend; "
            "its fused kernels are already CUDA Graph-capture compatible"
        )
        compile_user = False
    optimized = maybe_compile(optimized, compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    print(
        f"optimized_backend={optimized_backend} "
        f"(requested={args.optimized_backend})"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    if optimized_backend in ("cuda", "cuda-hybrid", "cuda-aggressive"):
        print(f"precision_plan={optimized.precision_plan.describe()}")
    if optimized_backend == "cuda-aggressive":
        print(
            f"cuda_graph={optimized.enable_cuda_graph}, "
            f"cuBLASLt_FFN={optimized.experimental_lt_ffn}, "
            f"forced_flash={args.force_flash_attention}"
        )

    if args.expanded_accuracy:
        if not custom_cuda_shape_supported(config, device, dtype):
            raise ValueError("expanded aggressive validation requires default CUDA FP32")
        print("\n=== Expanded 60-case accuracy matrix ===")
        matrix = ExpandedAccuracyHarness(
            config,
            device,
            args.accuracy_trials,
            args.seed,
            args.rtol,
            args.atol,
            use_lt_ffn,
        )
        matrix_result = matrix.evaluate(optimized.precision_plan, verbose=True)
        print(
            f"expanded summary: {'PASS' if matrix_result.passed else 'FAIL'} | "
            f"max_abs={matrix_result.max_abs_error:.6g} | "
            f"max_rel={matrix_result.max_relative_error:.6g} | "
            f"failed={matrix_result.failed_elements}/{matrix_result.total_elements}"
        )
        del matrix
        torch.cuda.empty_cache()
        if not matrix_result.passed and not args.benchmark_on_failure:
            return 2

    graph_passed = run_cuda_graph_replay_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
    )

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
        rtol=args.rtol,
        atol=args.atol,
    )

    accuracy_passed &= graph_passed
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


if __name__ == "__main__":
    raise SystemExit(main())
