#!/usr/bin/env python3
"""
Benchmark a user-optimized TensorFlow Transformer against a baseline
implementation using a compact, representative set of input dimensions.

Default values:
  batch_size: 1, 4, 16, 128, 10000
  qkv_dim:    32, 128, 1024
  heads:      1, 2, 4, 16
  seq_len:    32, 1024, 100000

The default case generator uses one-factor-at-a-time sweeps instead of a full
Cartesian product. Each configured value is covered while the other dimensions
stay at representative defaults. Because full attention grows quadratically in
sequence length, non-sequence sweeps use the shortest configured sequence. The
longest sequence uses batch_size=32 with the largest QKV/head configuration.

For every output element, correctness requires:
  abs(user - baseline) < atol
  OR
  abs(user - baseline) <= rtol * abs(baseline)

The default thresholds are atol=0.002 and rtol=0.02 (2%). Only cases that pass
correctness validation are benchmarked. Results are written to Markdown.

Requires TensorFlow 2.x and NumPy.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Transformer implementations
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    @property
    def qkv_dim(self) -> int:
        # In this benchmark, Q/K/V projections all use d_model as their output
        # dimension, so qkv_dim and d_model are identical.
        return self.d_model

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("qkv_dim/d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"qkv_dim ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(tf.keras.layers.Layer):
    """Reference multi-head self-attention using native TensorFlow operators."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dtype: tf.dtypes.DType,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(dtype=dtype, name=name)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = tf.keras.layers.Dense(
            d_model, use_bias=True, dtype=dtype, name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            d_model, use_bias=True, dtype=dtype, name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            d_model, use_bias=True, dtype=dtype, name="v_proj"
        )
        self.out_proj = tf.keras.layers.Dense(
            d_model, use_bias=True, dtype=dtype, name="out_proj"
        )

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(x)
        batch_size = shape[0]
        seq_len = shape[1]
        x = tf.reshape(
            x, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        return tf.transpose(x, [0, 2, 1, 3])

    def call(
        self,
        x: tf.Tensor,
        valid_token_mask: Optional[tf.Tensor] = None,
        causal: bool = False,
        training: bool = False,
    ) -> tf.Tensor:
        del training
        shape = tf.shape(x)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores * tf.cast(self.scale, scores.dtype)
        negative_infinity = tf.cast(float("-inf"), scores.dtype)

        if causal:
            row = tf.range(seq_len)[:, None]
            col = tf.range(seq_len)[None, :]
            future_mask = col > row
            scores = tf.where(future_mask[None, None, :, :], negative_infinity, scores)

        if valid_token_mask is not None:
            invalid_keys = tf.logical_not(valid_token_mask)[:, None, None, :]
            scores = tf.where(invalid_keys, negative_infinity, scores)

        # Stable reference: softmax accumulation is explicitly performed in fp32.
        probabilities = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        probabilities = tf.cast(probabilities, x.dtype)

        context = tf.matmul(probabilities, v)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, seq_len, self.d_model])
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = tf.where(valid_token_mask[..., None], output, tf.zeros_like(output))
        return output


class BaselineTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dtype: tf.dtypes.DType,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(dtype=dtype, name=name)
        self.norm1 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, dtype=dtype, name="norm1"
        )
        self.attention = BaselineSelfAttention(
            d_model, num_heads, dtype=dtype, name="attention"
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, dtype=dtype, name="norm2"
        )
        self.ffn_in = tf.keras.layers.Dense(
            ffn_dim, use_bias=True, dtype=dtype, name="ffn_in"
        )
        self.ffn_out = tf.keras.layers.Dense(
            d_model, use_bias=True, dtype=dtype, name="ffn_out"
        )

    def call(
        self,
        x: tf.Tensor,
        valid_token_mask: Optional[tf.Tensor],
        causal: bool,
        training: bool = False,
    ) -> tf.Tensor:
        attention_output = self.attention(
            self.norm1(x),
            valid_token_mask=valid_token_mask,
            causal=causal,
            training=training,
        )
        x = x + attention_output
        hidden = self.ffn_in(self.norm2(x))
        hidden = tf.nn.gelu(hidden, approximate=False)
        x = x + self.ffn_out(hidden)
        if valid_token_mask is not None:
            x = tf.where(valid_token_mask[..., None], x, tf.zeros_like(x))
        return x


class BaselineTransformer(tf.keras.Model):
    def __init__(
        self,
        config: TransformerConfig,
        dtype: tf.dtypes.DType,
        name: str = "baseline_transformer",
    ) -> None:
        super().__init__(dtype=dtype, name=name)
        self.config = config
        self.blocks = [
            BaselineTransformerBlock(
                config.d_model,
                config.num_heads,
                config.ffn_dim,
                dtype=dtype,
                name=f"layer_{index}",
            )
            for index in range(config.num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, dtype=dtype, name="final_norm"
        )

    def call(
        self,
        x: tf.Tensor,
        valid_token_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        for block in self.blocks:
            x = block(
                x,
                valid_token_mask=valid_token_mask,
                causal=self.config.causal,
                training=training,
            )
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = tf.where(valid_token_mask[..., None], x, tf.zeros_like(x))
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """
    Replace this class with the optimized implementation.

    Requirements:
      1. Keep the call signature unchanged.
      2. Return [batch_size, seq_len, qkv_dim].
      3. Keep compatible parameter order/names, or customize copy_model_weights().
    """

    def call(
        self,
        x: tf.Tensor,
        valid_token_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # ====================== your codes here ======================
        # Possible optimization directions:
        #   * fused/custom TensorFlow ops
        #   * tf.function(jit_compile=True) / XLA
        #   * Flash-Attention style kernels
        #   * fused LayerNorm / residual / FFN
        #
        # The fallback below makes the benchmark runnable before an optimized
        # implementation is inserted.
        return super().call(x, valid_token_mask, training=training)
        # ============================================================


def build_model(model: tf.keras.Model, config: TransformerConfig, dtype: tf.dtypes.DType) -> None:
    """Create model variables with a tiny input rather than a full benchmark case."""
    dummy_x = tf.zeros([1, 1, config.d_model], dtype=dtype)
    dummy_mask = tf.ones([1, 1], dtype=tf.bool)
    _ = model(dummy_x, dummy_mask, training=False)


def _variable_key(variable: tf.Variable) -> str:
    # Keras 3 variables may expose .path; older tf.keras variables expose .name.
    raw = str(getattr(variable, "path", variable.name)).split(":", maxsplit=1)[0]
    parts = raw.split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else parts[0]


def copy_model_weights(
    baseline: tf.keras.Model,
    optimized: tf.keras.Model,
    strict: bool = True,
) -> None:
    if strict:
        baseline_weights = baseline.get_weights()
        optimized_weights = optimized.get_weights()
        if len(baseline_weights) != len(optimized_weights):
            raise ValueError(
                "weight count mismatch: "
                f"baseline={len(baseline_weights)}, optimized={len(optimized_weights)}"
            )
        for index, (source, target) in enumerate(
            zip(baseline_weights, optimized_weights)
        ):
            if source.shape != target.shape:
                raise ValueError(
                    f"weight shape mismatch at index {index}: "
                    f"baseline={source.shape}, optimized={target.shape}"
                )
        optimized.set_weights(baseline_weights)
        return

    source_by_key = {_variable_key(v): v for v in baseline.weights}
    copied = 0
    missing: List[str] = []
    mismatched: List[str] = []
    for target in optimized.weights:
        key = _variable_key(target)
        source = source_by_key.get(key)
        if source is None:
            missing.append(key)
            continue
        if tuple(source.shape) != tuple(target.shape):
            mismatched.append(
                f"{key}: baseline={tuple(source.shape)}, optimized={tuple(target.shape)}"
            )
            continue
        target.assign(tf.cast(source, target.dtype))
        copied += 1

    print(f"[warning] non-strict weight copy assigned {copied} variables")
    if missing:
        print(f"[warning] optimized variables without source: {missing}")
    if mismatched:
        print(f"[warning] mismatched variable shapes: {mismatched}")


# -----------------------------------------------------------------------------
# Device, dtype, and runtime configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    index: int
    tf_name: str
    description: str

    @property
    def is_gpu(self) -> bool:
        return self.kind == "gpu"


def _parse_device_index(device_arg: str) -> int:
    if ":" not in device_arg:
        return 0
    suffix = device_arg.rsplit(":", maxsplit=1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid device index in {device_arg!r}") from exc


def resolve_device(device_arg: str) -> DeviceSpec:
    normalized = device_arg.strip().lower()
    gpus = tf.config.list_physical_devices("GPU")

    if normalized == "auto":
        normalized = "gpu:0" if gpus else "cpu:0"
    elif normalized in {"cuda", "gpu"}:
        normalized = "gpu:0"
    elif normalized.startswith("cuda:"):
        normalized = "gpu:" + normalized.split(":", maxsplit=1)[1]
    elif normalized == "cpu":
        normalized = "cpu:0"

    if normalized.startswith("gpu:"):
        index = _parse_device_index(normalized)
        if index < 0 or index >= len(gpus):
            raise RuntimeError(
                f"GPU {index} was requested, but TensorFlow sees {len(gpus)} GPU(s)"
            )
        details = tf.config.experimental.get_device_details(gpus[index])
        name = str(details.get("device_name", gpus[index].name))
        return DeviceSpec("gpu", index, f"/GPU:{index}", name)

    if normalized.startswith("cpu:"):
        index = _parse_device_index(normalized)
        return DeviceSpec("cpu", index, f"/CPU:{index}", f"CPU:{index}")

    raise ValueError("device must be auto, cpu, cpu:0, gpu, gpu:0, cuda, or cuda:0")


def configure_tensorflow_runtime(
    requested_device: str,
    memory_growth: bool,
    synchronous_execution: bool,
    allow_tf32: bool,
) -> None:
    """Configure GPU allocation and execution mode before creating tensors."""
    gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "TensorFlow runtime was initialized before memory growth "
                    "could be configured"
                ) from exc

    # Accurate per-inference timing requires each call to finish before the
    # timer stops. This avoids measuring only asynchronous GPU dispatch time.
    tf.config.experimental.set_synchronous_execution(synchronous_execution)
    tf.config.experimental.enable_tensor_float_32_execution(allow_tf32)

    # Validate the requested syntax early even if actual resolution happens next.
    normalized = requested_device.strip().lower()
    if normalized not in {"auto", "cpu", "gpu", "cuda"} and not any(
        normalized.startswith(prefix) for prefix in ("cpu:", "gpu:", "cuda:")
    ):
        raise ValueError(f"unsupported device: {requested_device}")


def resolve_dtype(dtype_name: str) -> tf.dtypes.DType:
    return {
        "float32": tf.float32,
        "float16": tf.float16,
        "bfloat16": tf.bfloat16,
    }[dtype_name]


def dtype_nbytes(dtype: tf.dtypes.DType) -> int:
    return int(dtype.size)


def make_model_runner(
    model: tf.keras.Model,
    config: TransformerConfig,
    dtype: tf.dtypes.DType,
    use_tf_function: bool,
    jit_compile: bool,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if not use_tf_function:
        if jit_compile:
            raise ValueError("XLA compilation requires tf.function graph mode")

        def eager_runner(x: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
            return model(x, valid_mask, training=False)

        return eager_runner

    input_signature = [
        tf.TensorSpec(
            [config.batch_size, config.seq_len, config.d_model],
            dtype=dtype,
            name="x",
        ),
        tf.TensorSpec(
            [config.batch_size, config.seq_len],
            dtype=tf.bool,
            name="valid_token_mask",
        ),
    ]

    @tf.function(
        input_signature=input_signature,
        reduce_retracing=True,
        jit_compile=jit_compile,
    )
    def graph_runner(x: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
        return model(x, valid_mask, training=False)

    return graph_runner


# -----------------------------------------------------------------------------
# Data generation and numerical validation
# -----------------------------------------------------------------------------


def _stateless_seed(seed: int, stream: int) -> tf.Tensor:
    # TensorFlow stateless RNG expects two signed int32 values.
    first = int(seed) & 0x7FFFFFFF
    second = int(stream) & 0x7FFFFFFF
    return tf.constant([first, second], dtype=tf.int32)


def generate_random_case(
    config: TransformerConfig,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    with tf.device(device.tf_name):
        x = tf.random.stateless_normal(
            [config.batch_size, config.seq_len, config.d_model],
            seed=_stateless_seed(seed, 1),
            dtype=dtype,
        )
        x = x * tf.cast(input_scale, dtype)

        if padding_ratio <= 0:
            valid_token_mask = tf.ones(
                [config.batch_size, config.seq_len], dtype=tf.bool
            )
            return x, valid_token_mask

        min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
        lengths = tf.random.stateless_uniform(
            [config.batch_size],
            seed=_stateless_seed(seed, 2),
            minval=min_valid,
            maxval=config.seq_len + 1,
            dtype=tf.int32,
        )
        valid_token_mask = tf.sequence_mask(lengths, maxlen=config.seq_len)
        x = tf.where(valid_token_mask[..., None], x, tf.zeros_like(x))
        return x, valid_token_mask


@dataclass
class AccuracySummary:
    passed: bool
    trials: int
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def unravel_index(flat_index: int, shape: Sequence[int]) -> Tuple[int, ...]:
    result: List[int] = []
    remaining = flat_index
    for size in reversed(shape):
        result.append(remaining % int(size))
        remaining //= int(size)
    return tuple(reversed(result))


def compare_outputs_chunked(
    reference: tf.Tensor,
    optimized: tf.Tensor,
    rtol: float,
    atol: float,
    chunk_elements: int,
) -> AccuracySummary:
    """Compare large outputs in host chunks to cap temporary fp32 memory."""
    reference_shape_list = reference.shape.as_list()
    optimized_shape_list = optimized.shape.as_list()
    if any(dim is None for dim in reference_shape_list):
        reference_shape = tuple(int(value) for value in tf.shape(reference).numpy())
    else:
        reference_shape = tuple(int(dim) for dim in reference_shape_list)
    if any(dim is None for dim in optimized_shape_list):
        optimized_shape = tuple(int(value) for value in tf.shape(optimized).numpy())
    else:
        optimized_shape = tuple(int(dim) for dim in optimized_shape_list)
    if reference_shape != optimized_shape:
        raise AssertionError(
            f"shape mismatch: baseline={reference_shape}, optimized={optimized_shape}"
        )

    ref_flat = tf.reshape(reference, [-1])
    opt_flat = tf.reshape(optimized, [-1])
    total = math.prod(reference_shape)

    failed_total = 0
    max_abs = 0.0
    max_rel = 0.0
    abs_sum = 0.0
    worst_flat_index = 0
    worst_ref = 0.0
    worst_opt = 0.0

    for start in range(0, total, chunk_elements):
        end = min(start + chunk_elements, total)
        ref = tf.cast(ref_flat[start:end], tf.float32).numpy()
        opt = tf.cast(opt_flat[start:end], tf.float32).numpy()

        finite = np.isfinite(ref) & np.isfinite(opt)
        abs_error = np.abs(opt - ref)
        abs_ok = abs_error < atol
        rel_ok = abs_error <= rtol * np.abs(ref)
        passed = finite & (abs_ok | rel_ok)
        failed_total += int(np.count_nonzero(~passed))

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_error = abs_error / np.maximum(np.abs(ref), 1e-12)
        rel_error = np.nan_to_num(
            rel_error, nan=np.inf, posinf=np.inf, neginf=np.inf
        )
        max_rel = max(max_rel, float(np.max(rel_error, initial=0.0)))

        safe_abs_error = np.nan_to_num(
            abs_error, nan=np.inf, posinf=np.inf, neginf=np.inf
        )
        abs_sum += float(np.sum(safe_abs_error, dtype=np.float64))
        local_index = int(np.argmax(safe_abs_error)) if safe_abs_error.size else 0
        chunk_max_abs = (
            float(safe_abs_error[local_index]) if safe_abs_error.size else 0.0
        )
        if chunk_max_abs > max_abs or start == 0:
            max_abs = chunk_max_abs
            worst_flat_index = start + local_index
            worst_ref = float(ref[local_index]) if ref.size else 0.0
            worst_opt = float(opt[local_index]) if opt.size else 0.0

    return AccuracySummary(
        passed=failed_total == 0,
        trials=1,
        total_elements=total,
        failed_elements=failed_total,
        max_abs_error=max_abs,
        max_relative_error=max_rel,
        mean_abs_error=abs_sum / max(total, 1),
        worst_index=unravel_index(worst_flat_index, reference_shape),
        reference_at_worst=worst_ref,
        optimized_at_worst=worst_opt,
    )


def run_accuracy_tests(
    baseline_runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    optimized_runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    config: TransformerConfig,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
    compare_chunk_elements: int,
) -> AccuracySummary:
    all_passed = True
    total_elements = 0
    failed_elements = 0
    max_abs = 0.0
    max_rel = 0.0
    weighted_abs_sum = 0.0
    worst_index: Tuple[int, ...] = ()
    worst_ref = 0.0
    worst_opt = 0.0

    for trial in range(trials):
        x, valid_mask = generate_random_case(
            config=config,
            device=device,
            dtype=dtype,
            seed=seed + trial,
            padding_ratio=padding_ratio,
            input_scale=input_scale,
        )
        reference = baseline_runner(x, valid_mask)
        candidate = optimized_runner(x, valid_mask)
        result = compare_outputs_chunked(
            reference,
            candidate,
            rtol=rtol,
            atol=atol,
            chunk_elements=compare_chunk_elements,
        )

        all_passed = all_passed and result.passed
        total_elements += result.total_elements
        failed_elements += result.failed_elements
        weighted_abs_sum += result.mean_abs_error * result.total_elements
        max_rel = max(max_rel, result.max_relative_error)
        if result.max_abs_error >= max_abs:
            max_abs = result.max_abs_error
            worst_index = result.worst_index
            worst_ref = result.reference_at_worst
            worst_opt = result.optimized_at_worst

        del x, valid_mask, reference, candidate

    return AccuracySummary(
        passed=all_passed,
        trials=trials,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=max_abs,
        max_relative_error=max_rel,
        mean_abs_error=weighted_abs_sum / max(total_elements, 1),
        worst_index=worst_index,
        reference_at_worst=worst_ref,
        optimized_at_worst=worst_opt,
    )


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------


def percentile(values: List[float], q: float) -> float:
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
    runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    x: tf.Tensor,
    valid_mask: tf.Tensor,
    iterations: int,
) -> None:
    output: Optional[tf.Tensor] = None
    for _ in range(iterations):
        output = runner(x, valid_mask)
    # The benchmark configures synchronous execution. Materializing one scalar
    # also provides a conservative barrier if a runtime ignores that setting.
    if output is not None:
        _ = tf.reduce_sum(tf.cast(output, tf.float32)).numpy()


def benchmark_once(
    runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    x: tf.Tensor,
    valid_mask: tf.Tensor,
    iterations: int,
) -> List[float]:
    samples_ms: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = runner(x, valid_mask)
        end = time.perf_counter_ns()
        samples_ms.append((end - start) / 1e6)
    # Retain and materialize the final result so the computation cannot be
    # eliminated and any outstanding work is complete.
    _ = tf.reduce_sum(tf.cast(output, tf.float32)).numpy()
    return samples_ms


@dataclass
class BenchmarkSummary:
    baseline: TimingResult
    optimized: TimingResult
    speedup: float
    baseline_tokens_per_second: float
    optimized_tokens_per_second: float


def benchmark_models(
    baseline_runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    optimized_runner: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    config: TransformerConfig,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> BenchmarkSummary:
    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    warmup_model(baseline_runner, x, valid_mask, warmup)
    warmup_model(optimized_runner, x, valid_mask, warmup)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate order between rounds to reduce clock/temperature order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline_runner, x, valid_mask, repeats)
            )
            optimized_samples.extend(
                benchmark_once(optimized_runner, x, valid_mask, repeats)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized_runner, x, valid_mask, repeats)
            )
            baseline_samples.extend(
                benchmark_once(baseline_runner, x, valid_mask, repeats)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens = config.batch_size * config.seq_len

    del x, valid_mask
    return BenchmarkSummary(
        baseline=baseline_result,
        optimized=optimized_result,
        speedup=speedup,
        baseline_tokens_per_second=tokens * 1000.0 / baseline_result.median_ms,
        optimized_tokens_per_second=tokens * 1000.0 / optimized_result.median_ms,
    )


# -----------------------------------------------------------------------------
# Matrix runner, feasibility checks, and Markdown report
# -----------------------------------------------------------------------------


@dataclass
class CaseResult:
    config: TransformerConfig
    status: str
    reason: str = ""
    estimated_peak_gib: float = 0.0
    accuracy: Optional[AccuracySummary] = None
    benchmark: Optional[BenchmarkSummary] = None


def _system_available_memory_bytes() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, ValueError, OSError):
        return 8 * 1024**3


def _nvidia_smi_free_memory_bytes(gpu_index: int) -> Optional[int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                f"--id={gpu_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        free_mib = int(first_line.strip())
        return free_mib * 1024**2
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def available_memory_bytes(device: DeviceSpec) -> int:
    if device.is_gpu:
        free = _nvidia_smi_free_memory_bytes(device.index)
        if free is not None:
            return free
        # Conservative fallback when the driver tool is unavailable.
        return 8 * 1024**3
    return _system_available_memory_bytes()


def estimate_baseline_peak_bytes(
    config: TransformerConfig,
    dtype: tf.dtypes.DType,
) -> int:
    """
    Conservative estimate for this explicit-attention baseline.

    The dominant term is [B, H, S, S]. Because reference softmax is evaluated
    in fp32, several score/probability buffers may overlap. The estimate is not
    an exact allocator trace; it is a safety guard against obviously impossible
    configurations.
    """
    b = config.batch_size
    s = config.seq_len
    d = config.d_model
    h = config.num_heads
    f = config.ffn_dim
    e = dtype_nbytes(dtype)

    # Two models are resident simultaneously. Approximate parameters per layer:
    # Q/K/V/out = 4*d*d; FFN = 2*d*f; norms/biases are lower order.
    params_per_model = config.num_layers * (4 * d * d + 2 * d * f) + 2 * d
    model_bytes = 2 * params_per_model * e

    # Input/output/residual/QKV/FFN temporaries for one active model.
    token_elements = b * s * d
    ffn_elements = b * s * f
    token_workspace = (10 * token_elements + 2 * ffn_elements) * e

    # Explicit scores plus fp32 softmax intermediates and casted probabilities.
    score_elements = b * h * s * s
    attention_workspace = score_elements * (2 * e + 8)

    mask_bytes = b * s + (s * s if config.causal else 0)
    return int(model_bytes + token_workspace + attention_workspace + mask_bytes)


def format_gib(value_bytes: int) -> float:
    return value_bytes / 1024**3


def cleanup_case(device: DeviceSpec) -> None:
    del device
    gc.collect()
    tf.keras.backend.clear_session()


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, tf.errors.ResourceExhaustedError) or any(
        marker in text
        for marker in (
            "out of memory",
            "resource exhausted",
            "oom when allocating",
            "failed to allocate memory",
        )
    )


def run_one_case(
    config: TransformerConfig,
    args: argparse.Namespace,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
    case_index: int,
    total_cases: int,
) -> CaseResult:
    estimated_bytes = estimate_baseline_peak_bytes(config, dtype)
    estimated_gib = format_gib(estimated_bytes)
    free_bytes = available_memory_bytes(device)
    budget_bytes = min(
        int(free_bytes * args.memory_fraction),
        int(args.max_estimated_memory_gib * 1024**3),
    )

    label = (
        f"[{case_index:02d}/{total_cases}] B={config.batch_size}, "
        f"S={config.seq_len}, QKV={config.qkv_dim}, H={config.num_heads}"
    )
    print(f"\n{label} | estimated peak={estimated_gib:.2f} GiB")

    if estimated_bytes > budget_bytes and not args.force_large_cases:
        reason = (
            f"estimated peak {estimated_gib:.2f} GiB exceeds execution budget "
            f"{format_gib(budget_bytes):.2f} GiB"
        )
        print(f"  SKIPPED: {reason}")
        return CaseResult(
            config=config,
            status="SKIPPED",
            reason=reason,
            estimated_peak_gib=estimated_gib,
        )

    baseline: Optional[tf.keras.Model] = None
    optimized: Optional[tf.keras.Model] = None
    baseline_runner: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None
    optimized_runner: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None
    try:
        case_seed = args.seed + case_index * 1009
        tf.keras.utils.set_random_seed(case_seed)

        with tf.device(device.tf_name):
            baseline = BaselineTransformer(
                config, dtype=dtype, name="baseline_transformer"
            )
            optimized = UserOptimizedTransformer(
                config, dtype=dtype, name="optimized_transformer"
            )
            build_model(baseline, config, dtype)
            build_model(optimized, config, dtype)
            copy_model_weights(
                baseline,
                optimized,
                strict=not args.non_strict_weight_copy,
            )

            baseline_runner = make_model_runner(
                baseline,
                config,
                dtype,
                use_tf_function=not args.eager,
                jit_compile=args.compile_baseline,
            )
            optimized_runner = make_model_runner(
                optimized,
                config,
                dtype,
                use_tf_function=not args.eager,
                jit_compile=args.compile_user,
            )

        accuracy = run_accuracy_tests(
            baseline_runner=baseline_runner,
            optimized_runner=optimized_runner,
            config=config,
            device=device,
            dtype=dtype,
            trials=args.accuracy_trials,
            seed=args.seed + case_index * 100000,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            rtol=args.rtol,
            atol=args.atol,
            compare_chunk_elements=args.compare_chunk_elements,
        )

        if not accuracy.passed:
            reason = (
                f"accuracy failed: {accuracy.failed_elements}/"
                f"{accuracy.total_elements} elements"
            )
            print(
                f"  FAIL accuracy | max_abs={accuracy.max_abs_error:.6g} | "
                f"max_rel={accuracy.max_relative_error:.6g} | {reason}"
            )
            return CaseResult(
                config=config,
                status="FAIL",
                reason=reason,
                estimated_peak_gib=estimated_gib,
                accuracy=accuracy,
            )

        print(
            f"  PASS accuracy | max_abs={accuracy.max_abs_error:.6g} | "
            f"max_rel={accuracy.max_relative_error:.6g}"
        )

        benchmark = benchmark_models(
            baseline_runner=baseline_runner,
            optimized_runner=optimized_runner,
            config=config,
            device=device,
            dtype=dtype,
            seed=args.seed + case_index * 100000,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.benchmark_rounds,
        )
        print(
            f"  benchmark | baseline={benchmark.baseline.median_ms:.4f} ms | "
            f"optimized={benchmark.optimized.median_ms:.4f} ms | "
            f"speedup={benchmark.speedup:.3f}x"
        )
        return CaseResult(
            config=config,
            status="PASS",
            estimated_peak_gib=estimated_gib,
            accuracy=accuracy,
            benchmark=benchmark,
        )

    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        status = "OOM" if is_oom_error(exc) else "ERROR"
        reason = f"{type(exc).__name__}: {exc}"
        print(f"  {status}: {reason}")
        if args.print_traceback and status == "ERROR":
            traceback.print_exc()
        return CaseResult(
            config=config,
            status=status,
            reason=reason,
            estimated_peak_gib=estimated_gib,
        )
    finally:
        del baseline_runner, optimized_runner, baseline, optimized
        cleanup_case(device)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown_report(
    output_path: Path,
    results: Sequence[CaseResult],
    args: argparse.Namespace,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
) -> None:
    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    skipped = sum(result.status == "SKIPPED" for result in results)
    oom = sum(result.status == "OOM" for result in results)
    errors = sum(result.status == "ERROR" for result in results)

    execution_mode = "eager" if args.eager else "tf.function"
    lines = [
        "# Transformer 精度与性能测试报告（TensorFlow）",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- TensorFlow：`{tf.__version__}`",
        f"- 设备：`{device.tf_name}`（{device.description}）",
        f"- 数据类型：`{dtype.name}`",
        f"- 执行模式：`{execution_mode}`",
        f"- Baseline XLA：`{args.compile_baseline}`",
        f"- Optimized XLA：`{args.compile_user}`",
        f"- TF32：`{args.allow_tf32}`",
        f"- Heads：`{', '.join(map(str, args.heads))}`",
        "- 测试策略：单变量扫描；每次改变一个维度，其余维度保持代表性默认值。",
        "- 非 seq_len 扫描使用最短序列长度；最长序列固定使用 batch_size=32，并使用最大的 qkv_dim 和 heads。若资源预估超限，该组合会在报表中标记为 SKIPPED。",
        f"- Transformer layers：`{args.layers}`",
        f"- FFN：`{args.ffn_dim if args.ffn_dim > 0 else f'{args.ffn_multiplier} × qkv_dim'}`",
        f"- 精度标准：逐元素满足 `abs_error < {args.atol:g}` 或 `abs_error <= {args.rtol:g} × abs(reference)`",
        f"- 性能统计：每组预热 `{args.warmup}` 次，`{args.benchmark_rounds}` 轮 × `{args.repeats}` 次，使用中位延迟计算加速比",
        "- 为正确测量 GPU 延迟，TensorFlow 操作使用同步执行模式。",
        "- `qkv_dim` 在本实现中同时作为 Transformer 的 `d_model`，Q/K/V 投影维度相同。",
        "",
        "## 汇总",
        "",
        f"共 `{len(results)}` 组：通过并完成测速 `{passed}`，精度失败 `{failed}`，资源预检跳过 `{skipped}`，运行时 OOM `{oom}`，其他错误 `{errors}`。",
        "",
        "## 分维度结果",
        "",
        "| batch_size | qkv_dim | heads | seq_len | 状态 | 最大绝对误差 | 最大相对误差 | Baseline 中位延迟 (ms) | Optimized 中位延迟 (ms) | 加速比 | 估算峰值 (GiB) | 说明 |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for result in results:
        accuracy = result.accuracy
        benchmark = result.benchmark
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.config.batch_size),
                    str(result.config.qkv_dim),
                    str(result.config.num_heads),
                    str(result.config.seq_len),
                    result.status,
                    format_float(
                        accuracy.max_abs_error if accuracy is not None else None, 6
                    ),
                    format_float(
                        accuracy.max_relative_error if accuracy is not None else None,
                        6,
                    ),
                    format_float(
                        benchmark.baseline.median_ms
                        if benchmark is not None
                        else None,
                        4,
                    ),
                    format_float(
                        benchmark.optimized.median_ms
                        if benchmark is not None
                        else None,
                        4,
                    ),
                    f"{benchmark.speedup:.3f}x" if benchmark is not None else "-",
                    f"{result.estimated_peak_gib:.2f}",
                    markdown_escape(result.reason),
                ]
            )
            + " |"
        )

    successful = [result for result in results if result.benchmark is not None]
    if successful:
        geometric_mean = math.exp(
            statistics.fmean(
                math.log(result.benchmark.speedup) for result in successful
            )
        )
        lines.extend(
            [
                "",
                "## 有效测试汇总",
                "",
                f"- 有效组合数：`{len(successful)}`",
                f"- 加速比几何平均值：`{geometric_mean:.3f}x`",
                f"- 最低加速比：`{min(result.benchmark.speedup for result in successful):.3f}x`",
                f"- 最高加速比：`{max(result.benchmark.speedup for result in successful):.3f}x`",
            ]
        )

    if skipped or oom:
        lines.extend(
            [
                "",
                "## 资源限制说明",
                "",
                "标准全量 Self-Attention 会显式生成 `[batch_size, heads, seq_len, seq_len]` 中间张量，空间复杂度为 `O(batch_size × heads × seq_len²)`。资源预检只用于避免明显不可执行的组合；使用 `--force-large-cases` 可以关闭该预检，但可能导致进程或设备 OOM。",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI and compact dimension generation
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compact dimension benchmark for baseline vs optimized "
            "TensorFlow Transformer"
        )
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 128, 10000],
    )
    parser.add_argument(
        "--qkv-dims",
        type=int,
        nargs="+",
        default=[32, 128, 1024],
        help="Q/K/V projection dimension; also used as d_model",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[32, 1024, 100000],
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 16],
        help="number of attention heads to test",
    )
    parser.add_argument("--ffn-dim", type=int, default=0)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cpu:0, gpu, gpu:0, cuda, or cuda:0",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--compare-chunk-elements",
        type=int,
        default=4_000_000,
        help="number of output elements compared per fp32 host chunk",
    )

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--benchmark-rounds", type=int, default=3)

    parser.add_argument(
        "--eager",
        action="store_true",
        help="run both models in eager mode instead of tf.function",
    )
    parser.add_argument(
        "--compile-baseline",
        action="store_true",
        help="enable XLA jit_compile for the baseline tf.function",
    )
    parser.add_argument(
        "--compile-user",
        action="store_true",
        help="enable XLA jit_compile for the optimized tf.function",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow TensorFloat-32 for supported float32 GPU operations",
    )
    parser.add_argument(
        "--memory-growth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable TensorFlow GPU memory growth",
    )

    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="maximum fraction of currently free device/system memory per case",
    )
    parser.add_argument(
        "--max-estimated-memory-gib",
        type=float,
        default=128.0,
        help="absolute safety cap used by the preflight estimate",
    )
    parser.add_argument(
        "--force-large-cases",
        action="store_true",
        help="attempt cases even when the preflight estimate exceeds the budget",
    )
    parser.add_argument("--print-traceback", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tensorflow_transformer_benchmark_report.md"),
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    device: DeviceSpec,
    dtype: tf.dtypes.DType,
) -> None:
    for name in ("batch_sizes", "qkv_dims", "seq_lens", "heads"):
        values = getattr(args, name)
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"--{name.replace('_', '-')} must contain positive values")
    if args.layers <= 0:
        raise ValueError("layers must be positive")
    invalid_pairs = [
        (dim, heads)
        for dim in args.qkv_dims
        for heads in args.heads
        if dim % heads != 0
    ]
    if invalid_pairs:
        details = ", ".join(
            f"qkv_dim={dim}, heads={heads}" for dim, heads in invalid_pairs
        )
        raise ValueError(f"qkv_dim must be divisible by heads; invalid pairs: {details}")
    if args.ffn_dim < 0 or args.ffn_multiplier <= 0:
        raise ValueError("ffn-dim must be >= 0 and ffn-multiplier must be positive")
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy-trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.compare_chunk_elements <= 0:
        raise ValueError("compare-chunk-elements must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("invalid timing arguments")
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in (0, 1]")
    if args.max_estimated_memory_gib <= 0:
        raise ValueError("max-estimated-memory-gib must be positive")
    if args.eager and (args.compile_baseline or args.compile_user):
        raise ValueError("--eager cannot be combined with XLA compile flags")
    if device.kind == "cpu" and dtype == tf.float16:
        print("[warning] float16 CPU kernels may be unsupported or very slow")


def _unique_sorted(values: Sequence[int]) -> List[int]:
    return sorted(set(values))


def _middle_value(values: Sequence[int]) -> int:
    ordered = _unique_sorted(values)
    return ordered[len(ordered) // 2]


def build_compact_dimensions(args: argparse.Namespace) -> List[Tuple[int, int, int, int]]:
    """
    Build a compact one-factor-at-a-time test set.

    Representative defaults use the middle batch/QKV/head values. The shortest
    sequence is used as the working sequence for non-sequence sweeps because
    explicit attention has O(seq_len^2) memory and compute cost. The largest
    sequence is paired with batch_size=32 and the largest QKV/head values as the
    designated stress case. Resource preflight may mark it as SKIPPED.
    """
    batch_sizes = _unique_sorted(args.batch_sizes)
    qkv_dims = _unique_sorted(args.qkv_dims)
    heads_values = _unique_sorted(args.heads)
    seq_lens = _unique_sorted(args.seq_lens)

    default_batch = _middle_value(batch_sizes)
    default_qkv = _middle_value(qkv_dims)
    default_heads = _middle_value(heads_values)
    working_seq = min(seq_lens)
    longest_seq = max(seq_lens)

    if default_qkv % default_heads != 0:
        compatible_heads = [head for head in heads_values if default_qkv % head == 0]
        if not compatible_heads:
            raise ValueError(
                f"no configured heads value divides default qkv_dim={default_qkv}"
            )
        default_heads = min(
            compatible_heads, key=lambda head: abs(head - default_heads)
        )

    dimensions: List[Tuple[int, int, int, int]] = []

    def add_case(batch_size: int, qkv_dim: int, heads: int, seq_len: int) -> None:
        case = (batch_size, qkv_dim, heads, seq_len)
        if qkv_dim % heads != 0:
            return
        if case not in dimensions:
            dimensions.append(case)

    # Batch-size sweep. This also includes the representative default case.
    for batch_size in batch_sizes:
        add_case(batch_size, default_qkv, default_heads, working_seq)

    # QKV-dimension sweep. Keep a compatible head count near the default.
    for qkv_dim in qkv_dims:
        compatible_heads = [head for head in heads_values if qkv_dim % head == 0]
        if not compatible_heads:
            continue
        heads = min(compatible_heads, key=lambda head: abs(head - default_heads))
        add_case(default_batch, qkv_dim, heads, working_seq)

    # Head-count sweep. Keep a compatible QKV dimension near the default.
    for heads in heads_values:
        compatible_dims = [dim for dim in qkv_dims if dim % heads == 0]
        if not compatible_dims:
            continue
        qkv_dim = min(compatible_dims, key=lambda dim: abs(dim - default_qkv))
        add_case(default_batch, qkv_dim, heads, working_seq)

    # Sequence-length sweep. The longest sequence is the designated stress case:
    # batch_size=32, while QKV dimension and heads use configured maxima.
    for seq_len in seq_lens:
        if seq_len == longest_seq and len(seq_lens) > 1:
            add_case(32, max(qkv_dims), max(heads_values), seq_len)
        else:
            add_case(default_batch, default_qkv, default_heads, seq_len)

    return dimensions


def main() -> int:
    args = parse_args()

    # Must be configured before TensorFlow initializes accelerator contexts.
    configure_tensorflow_runtime(
        requested_device=args.device,
        memory_growth=args.memory_growth,
        synchronous_execution=True,
        allow_tf32=args.allow_tf32,
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    validate_args(args, device, dtype)

    tf.keras.utils.set_random_seed(args.seed)
    dimensions = build_compact_dimensions(args)

    print("=== Transformer compact benchmark (TensorFlow) ===")
    print(
        f"cases={len(dimensions)}, device={device.tf_name}, dtype={dtype.name}, "
        f"tensorflow={tf.__version__}"
    )
    print(f"device_name={device.description}")
    print(
        f"correctness: abs_error < {args.atol:g} OR "
        f"abs_error <= {args.rtol:g} * abs(reference)"
    )

    results: List[CaseResult] = []
    for index, (batch_size, qkv_dim, heads, seq_len) in enumerate(
        dimensions, start=1
    ):
        ffn_dim = (
            args.ffn_dim
            if args.ffn_dim > 0
            else qkv_dim * args.ffn_multiplier
        )
        config = TransformerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=qkv_dim,
            num_heads=heads,
            ffn_dim=ffn_dim,
            num_layers=args.layers,
            causal=args.causal,
        )
        config.validate()
        result = run_one_case(
            config=config,
            args=args,
            device=device,
            dtype=dtype,
            case_index=index,
            total_cases=len(dimensions),
        )
        results.append(result)
        # Keep a useful partial report even if the process is interrupted later.
        write_markdown_report(args.report, results, args, device, dtype)

    write_markdown_report(args.report, results, args, device, dtype)
    print(f"\nMarkdown report written to: {args.report.resolve()}")

    failed = any(result.status in {"FAIL", "ERROR"} for result in results)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
