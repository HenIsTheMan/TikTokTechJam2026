#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

// This extension is intentionally specialized for the benchmark shape.
constexpr int kDModel = 512;
constexpr int kFfn = 2048;
constexpr int kNormThreads = 128;     // 4 warps x 32, 4 float values/thread.
constexpr int kGeluThreads = 256;     // 64 float4 values/block.
constexpr float kInvSqrt2 = 0.70710678118654752440f;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int mask = 16; mask; mask >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, mask);
  }
  return x;
}

// One block = one [512] token row.
//
// Compared with the old kernel:
//   * one synchronization instead of two;
//   * mean/inv-std are broadcast with a warp shuffle, not a second shared read;
//   * bias/weight loads stay vectorized;
//   * invalid rows never read residual/delta and are zeroed directly.
__global__ __launch_bounds__(kNormThreads, 2)
void residual_bias_layer_norm_512_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ delta,
    const float* __restrict__ bias,
    const float* __restrict__ weight,
    const float* __restrict__ norm_bias,
    const bool* __restrict__ valid_mask,
    float* __restrict__ residual_out,
    float* __restrict__ norm_out,
    float eps,
    bool zero_invalid_norm) {
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const bool valid = (valid_mask == nullptr) || valid_mask[row];
  const int vec = row * (kDModel / 4) + t;

  float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
  if (valid) {
    const float4 r = reinterpret_cast<const float4*>(residual)[vec];
    const float4 d = reinterpret_cast<const float4*>(delta)[vec];
    const float4 b = reinterpret_cast<const float4*>(bias)[t];
    x.x = r.x + d.x + b.x;
    x.y = r.y + d.y + b.y;
    x.z = r.z + d.z + b.z;
    x.w = r.w + d.w + b.w;
  }

  reinterpret_cast<float4*>(residual_out)[vec] = x;

  float sum = x.x + x.y + x.z + x.w;
  float sq  = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  sum = warp_sum(sum);
  sq  = warp_sum(sq);

  __shared__ float partial_sum[4];
  __shared__ float partial_sq[4];
  __shared__ float stats[2];

  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) {
    partial_sum[warp] = sum;
    partial_sq[warp] = sq;
  }
  __syncthreads();

  if (warp == 0) {
    float total_sum = lane < 4 ? partial_sum[lane] : 0.f;
    float total_sq  = lane < 4 ? partial_sq[lane] : 0.f;
    total_sum = warp_sum(total_sum);
    total_sq  = warp_sum(total_sq);
    if (lane == 0) {
      const float mean = total_sum * (1.0f / kDModel);
      const float var = fmaxf(total_sq * (1.0f / kDModel) - mean * mean, 0.f);
      stats[0] = mean;
      stats[1] = rsqrtf(var + eps);
    }
  }

  __syncthreads();
  const float mean = stats[0];
  const float inv_std = stats[1];

  const float4 w = reinterpret_cast<const float4*>(weight)[t];
  const float4 nb = reinterpret_cast<const float4*>(norm_bias)[t];
  float4 y;
  y.x = (x.x - mean) * inv_std * w.x + nb.x;
  y.y = (x.y - mean) * inv_std * w.y + nb.y;
  y.z = (x.z - mean) * inv_std * w.z + nb.z;
  y.w = (x.w - mean) * inv_std * w.w + nb.w;

  if (!valid && zero_invalid_norm) {
    y = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  reinterpret_cast<float4*>(norm_out)[vec] = y;
}

template <bool kDeltaHalf, bool kNormHalf>
__global__ __launch_bounds__(kNormThreads, 2)
void mixed_residual_bias_layer_norm_512_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ delta_float,
    const __half* __restrict__ delta_half,
    const float* __restrict__ bias,
    const float* __restrict__ weight,
    const float* __restrict__ norm_bias,
    const bool* __restrict__ valid_mask,
    float* __restrict__ residual_out,
    float* __restrict__ norm_float_out,
    __half* __restrict__ norm_half_out,
    float eps,
    bool zero_invalid_norm) {
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const bool valid = (valid_mask == nullptr) || valid_mask[row];
  const int vec = row * (kDModel / 4) + t;

  float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
  if (valid) {
    const float4 r = reinterpret_cast<const float4*>(residual)[vec];
    const float4 b = reinterpret_cast<const float4*>(bias)[t];
    float4 d;
    if constexpr (kDeltaHalf) {
      const int half2_index = vec * 2;
      const __half2 d01 = reinterpret_cast<const __half2*>(delta_half)[half2_index];
      const __half2 d23 =
          reinterpret_cast<const __half2*>(delta_half)[half2_index + 1];
      const float2 f01 = __half22float2(d01);
      const float2 f23 = __half22float2(d23);
      d = make_float4(f01.x, f01.y, f23.x, f23.y);
    } else {
      d = reinterpret_cast<const float4*>(delta_float)[vec];
    }
    x.x = r.x + d.x + b.x;
    x.y = r.y + d.y + b.y;
    x.z = r.z + d.z + b.z;
    x.w = r.w + d.w + b.w;
  }
  reinterpret_cast<float4*>(residual_out)[vec] = x;

  float sum = warp_sum(x.x + x.y + x.z + x.w);
  float sq = warp_sum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
  __shared__ float partial_sum[4];
  __shared__ float partial_sq[4];
  __shared__ float stats[2];
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) {
    partial_sum[warp] = sum;
    partial_sq[warp] = sq;
  }
  __syncthreads();
  if (warp == 0) {
    float total_sum = lane < 4 ? partial_sum[lane] : 0.f;
    float total_sq = lane < 4 ? partial_sq[lane] : 0.f;
    total_sum = warp_sum(total_sum);
    total_sq = warp_sum(total_sq);
    if (lane == 0) {
      const float mean = total_sum * (1.0f / kDModel);
      const float variance =
          fmaxf(total_sq * (1.0f / kDModel) - mean * mean, 0.f);
      stats[0] = mean;
      stats[1] = rsqrtf(variance + eps);
    }
  }
  __syncthreads();

  const float mean = stats[0];
  const float inv_std = stats[1];
  const float4 w = reinterpret_cast<const float4*>(weight)[t];
  const float4 b = reinterpret_cast<const float4*>(norm_bias)[t];
  float4 y;
  y.x = (x.x - mean) * inv_std * w.x + b.x;
  y.y = (x.y - mean) * inv_std * w.y + b.y;
  y.z = (x.z - mean) * inv_std * w.z + b.z;
  y.w = (x.w - mean) * inv_std * w.w + b.w;
  if (!valid && zero_invalid_norm) y = make_float4(0.f, 0.f, 0.f, 0.f);

  if constexpr (kNormHalf) {
    const int half2_index = vec * 2;
    reinterpret_cast<__half2*>(norm_half_out)[half2_index] =
        __floats2half2_rn(y.x, y.y);
    reinterpret_cast<__half2*>(norm_half_out)[half2_index + 1] =
        __floats2half2_rn(y.z, y.w);
  } else {
    reinterpret_cast<float4*>(norm_float_out)[vec] = y;
  }
}

// Final LayerNorm specializes the benchmark's 512-wide rows. The masked
// variant folds the final masked_fill into this kernel and avoids reading
// invalid input rows entirely.
template <bool kHasMask, bool kOutputHalf>
__global__ __launch_bounds__(kNormThreads, 2)
void layer_norm_512_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const bool* __restrict__ valid_mask,
    float* __restrict__ output,
    __half* __restrict__ half_output,
    float eps) {
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const int vec = row * (kDModel / 4) + t;

  if constexpr (kHasMask) {
    if (!valid_mask[row]) {
      if constexpr (kOutputHalf) {
        const int half2_index = vec * 2;
        reinterpret_cast<__half2*>(half_output)[half2_index] = __float2half2_rn(0.f);
        reinterpret_cast<__half2*>(half_output)[half2_index + 1] =
            __float2half2_rn(0.f);
      } else {
        reinterpret_cast<float4*>(output)[vec] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      return;
    }
  }

  const float4 x = reinterpret_cast<const float4*>(input)[vec];
  float sum = x.x + x.y + x.z + x.w;
  float sq = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  sum = warp_sum(sum);
  sq = warp_sum(sq);

  __shared__ float partial_sum[4];
  __shared__ float partial_sq[4];
  __shared__ float stats[2];
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) {
    partial_sum[warp] = sum;
    partial_sq[warp] = sq;
  }
  __syncthreads();

  if (warp == 0) {
    float total_sum = lane < 4 ? partial_sum[lane] : 0.f;
    float total_sq = lane < 4 ? partial_sq[lane] : 0.f;
    total_sum = warp_sum(total_sum);
    total_sq = warp_sum(total_sq);
    if (lane == 0) {
      const float mean = total_sum * (1.0f / kDModel);
      const float variance =
          fmaxf(total_sq * (1.0f / kDModel) - mean * mean, 0.f);
      stats[0] = mean;
      stats[1] = rsqrtf(variance + eps);
    }
  }
  __syncthreads();

  const float mean = stats[0];
  const float inv_std = stats[1];
  const float4 w = reinterpret_cast<const float4*>(weight)[t];
  const float4 b = reinterpret_cast<const float4*>(bias)[t];
  float4 y;
  y.x = (x.x - mean) * inv_std * w.x + b.x;
  y.y = (x.y - mean) * inv_std * w.y + b.y;
  y.z = (x.z - mean) * inv_std * w.z + b.z;
  y.w = (x.w - mean) * inv_std * w.w + b.w;
  if constexpr (kOutputHalf) {
    const int half2_index = vec * 2;
    reinterpret_cast<__half2*>(half_output)[half2_index] =
        __floats2half2_rn(y.x, y.y);
    reinterpret_cast<__half2*>(half_output)[half2_index + 1] =
        __floats2half2_rn(y.z, y.w);
  } else {
    reinterpret_cast<float4*>(output)[vec] = y;
  }
}

// One block owns one token row of width 2048.
// Each thread handles exactly four float4s => 16 scalar values.
// This removes the old global-index modulo operation entirely.
__global__ __launch_bounds__(kGeluThreads, 2)
void bias_gelu_2048_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int t = threadIdx.x;
  const int base_vec = row * (kFfn / 4) + t;
  const int bias_vec = t;
  const int stride = kGeluThreads;

  const float4* in4 = reinterpret_cast<const float4*>(input);
  float4* out4 = reinterpret_cast<float4*>(output);
  const float4* b4 = reinterpret_cast<const float4*>(bias);

#pragma unroll
  for (int i = 0; i < (kFfn / 4) / kGeluThreads; ++i) {
    const int v = base_vec + i * stride;
    const float4 a = in4[v];
    const float4 b = b4[bias_vec + i * stride];

    float4 r;
    const float x0 = a.x + b.x;
    const float x1 = a.y + b.y;
    const float x2 = a.z + b.z;
    const float x3 = a.w + b.w;
    r.x = 0.5f * x0 * (1.0f + erff(x0 * kInvSqrt2));
    r.y = 0.5f * x1 * (1.0f + erff(x1 * kInvSqrt2));
    r.z = 0.5f * x2 * (1.0f + erff(x2 * kInvSqrt2));
    r.w = 0.5f * x3 * (1.0f + erff(x3 * kInvSqrt2));
    out4[v] = r;
  }
}

inline void check_fp32_contiguous_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::ScalarType::Float,
              name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor launch_layer_norm_512(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor* valid_token_mask,
    double eps) {
  check_fp32_contiguous_cuda(input, "input");
  check_fp32_contiguous_cuda(weight, "weight");
  check_fp32_contiguous_cuda(bias, "bias");
  TORCH_CHECK(input.dim() == 3 && input.size(-1) == kDModel,
              "layer_norm requires [B, S, 512] input");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == kDModel &&
                  bias.dim() == 1 && bias.numel() == kDModel,
              "weight and bias must contain 512 elements");
  TORCH_CHECK(input.device() == weight.device() && input.device() == bias.device(),
              "device mismatch");

  const int64_t rows64 = input.numel() / kDModel;
  TORCH_CHECK(rows64 <= INT32_MAX, "input too large");
  if (valid_token_mask != nullptr) {
    TORCH_CHECK(valid_token_mask->is_cuda() && valid_token_mask->is_contiguous(),
                "valid_token_mask must be contiguous CUDA");
    TORCH_CHECK(valid_token_mask->scalar_type() == at::ScalarType::Bool,
                "valid_token_mask must be bool");
    TORCH_CHECK(valid_token_mask->numel() == rows64,
                "mask must contain one value per token");
    TORCH_CHECK(valid_token_mask->device() == input.device(), "mask device mismatch");
  }

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  if (rows64 == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int rows = static_cast<int>(rows64);
  if (valid_token_mask == nullptr) {
    layer_norm_512_kernel<false, false><<<rows, kNormThreads, 0, stream>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        nullptr, output.data_ptr<float>(), nullptr, static_cast<float>(eps));
  } else {
    layer_norm_512_kernel<true, false><<<rows, kNormThreads, 0, stream>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        valid_token_mask->data_ptr<bool>(), output.data_ptr<float>(), nullptr,
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor layer_norm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
  return launch_layer_norm_512(input, weight, bias, nullptr, eps);
}

torch::Tensor layer_norm_mask_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    double eps) {
  return launch_layer_norm_512(input, weight, bias, &valid_token_mask, eps);
}

torch::Tensor layer_norm_half_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
  check_fp32_contiguous_cuda(input, "input");
  check_fp32_contiguous_cuda(weight, "weight");
  check_fp32_contiguous_cuda(bias, "bias");
  TORCH_CHECK(input.dim() == 3 && input.size(-1) == kDModel,
              "layer_norm_half requires [B, S, 512]");
  TORCH_CHECK(weight.numel() == kDModel && bias.numel() == kDModel,
              "weight and bias must contain 512 elements");
  TORCH_CHECK(input.device() == weight.device() && input.device() == bias.device(),
              "device mismatch");
  const int64_t rows64 = input.numel() / kDModel;
  TORCH_CHECK(rows64 <= INT32_MAX, "input too large");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      input.sizes(), input.options().dtype(at::ScalarType::Half));
  if (rows64 == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  layer_norm_512_kernel<false, true>
      <<<static_cast<int>(rows64), kNormThreads, 0, stream>>>(
          input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
          nullptr, nullptr,
          reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
          static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> mixed_residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool norm_output_half,
    bool zero_invalid_norm) {
  check_fp32_contiguous_cuda(residual, "residual");
  check_fp32_contiguous_cuda(bias, "bias");
  check_fp32_contiguous_cuda(weight, "weight");
  check_fp32_contiguous_cuda(norm_bias, "norm_bias");
  TORCH_CHECK(delta.is_cuda() && delta.is_contiguous(),
              "delta must be contiguous CUDA");
  const bool delta_half = delta.scalar_type() == at::ScalarType::Half;
  TORCH_CHECK(delta_half || delta.scalar_type() == at::ScalarType::Float,
              "delta must be float16 or float32");
  TORCH_CHECK(residual.sizes() == delta.sizes(), "residual/delta shape mismatch");
  TORCH_CHECK(residual.dim() == 3 && residual.size(-1) == kDModel,
              "requires [B, S, 512]");
  TORCH_CHECK(bias.numel() == kDModel && weight.numel() == kDModel &&
                  norm_bias.numel() == kDModel,
              "512-element parameters required");
  TORCH_CHECK(residual.device() == delta.device() &&
                  residual.device() == bias.device() &&
                  residual.device() == weight.device() &&
                  residual.device() == norm_bias.device(),
              "device mismatch");

  const int64_t rows64 = residual.numel() / kDModel;
  TORCH_CHECK(rows64 <= INT32_MAX, "input too large");
  const bool has_mask = valid_token_mask.numel() != 0;
  if (has_mask) {
    TORCH_CHECK(valid_token_mask.is_cuda() && valid_token_mask.is_contiguous() &&
                    valid_token_mask.scalar_type() == at::ScalarType::Bool,
                "valid_token_mask must be contiguous bool CUDA");
    TORCH_CHECK(valid_token_mask.numel() == rows64, "mask size mismatch");
  }

  c10::cuda::CUDAGuard device_guard(residual.device());
  auto residual_out = torch::empty_like(residual);
  auto norm_out = norm_output_half
      ? torch::empty(residual.sizes(), residual.options().dtype(at::ScalarType::Half))
      : torch::empty_like(residual);
  if (rows64 == 0) return {residual_out, norm_out};
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  const int rows = static_cast<int>(rows64);
  const float* delta_float = delta_half ? nullptr : delta.data_ptr<float>();
  const __half* delta_half_ptr = delta_half
      ? reinterpret_cast<const __half*>(delta.data_ptr<at::Half>()) : nullptr;
  const bool* mask = has_mask ? valid_token_mask.data_ptr<bool>() : nullptr;
#define LAUNCH_MIXED(DELTA_HALF, NORM_HALF) \
  mixed_residual_bias_layer_norm_512_kernel<DELTA_HALF, NORM_HALF> \
      <<<rows, kNormThreads, 0, stream>>>( \
          residual.data_ptr<float>(), delta_float, delta_half_ptr, \
          bias.data_ptr<float>(), weight.data_ptr<float>(), \
          norm_bias.data_ptr<float>(), mask, residual_out.data_ptr<float>(), \
          NORM_HALF ? nullptr : norm_out.data_ptr<float>(), \
          NORM_HALF ? reinterpret_cast<__half*>(norm_out.data_ptr<at::Half>()) : nullptr, \
          static_cast<float>(eps), zero_invalid_norm)
  if (delta_half && norm_output_half) {
    LAUNCH_MIXED(true, true);
  } else if (delta_half) {
    LAUNCH_MIXED(true, false);
  } else if (norm_output_half) {
    LAUNCH_MIXED(false, true);
  } else {
    LAUNCH_MIXED(false, false);
  }
#undef LAUNCH_MIXED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual_out, norm_out};
}

torch::Tensor bias_gelu_cuda(const torch::Tensor& input, const torch::Tensor& bias) {
  check_fp32_contiguous_cuda(input, "input");
  check_fp32_contiguous_cuda(bias, "bias");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == kFfn,
              "bias_gelu requires last dimension 2048");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == kFfn,
              "bias must contain 2048 elements");
  TORCH_CHECK(input.device() == bias.device(), "device mismatch");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t rows64 = input.numel() / kFfn;
  if (rows64 == 0) return output;
  TORCH_CHECK(rows64 <= INT32_MAX, "input too large");

  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  bias_gelu_2048_kernel<<<static_cast<int>(rows64), kGeluThreads, 0, stream>>>(
      input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(rows64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool zero_invalid_norm) {
  check_fp32_contiguous_cuda(residual, "residual");
  check_fp32_contiguous_cuda(delta, "delta");
  check_fp32_contiguous_cuda(bias, "bias");
  check_fp32_contiguous_cuda(weight, "weight");
  check_fp32_contiguous_cuda(norm_bias, "norm_bias");

  TORCH_CHECK(residual.sizes() == delta.sizes(), "residual/delta shape mismatch");
  TORCH_CHECK(residual.dim() == 3 && residual.size(-1) == kDModel,
              "requires [B, S, 512]");
  TORCH_CHECK(bias.numel() == kDModel && weight.numel() == kDModel &&
              norm_bias.numel() == kDModel,
              "512-element parameters required");
  TORCH_CHECK(residual.device() == delta.device() &&
              residual.device() == bias.device() &&
              residual.device() == weight.device() &&
              residual.device() == norm_bias.device(),
              "device mismatch");

  const int64_t rows64 = residual.numel() / kDModel;
  TORCH_CHECK(rows64 <= INT32_MAX, "input too large");

  const bool has_mask = valid_token_mask.numel() != 0;
  if (has_mask) {
    TORCH_CHECK(valid_token_mask.is_cuda() && valid_token_mask.is_contiguous(),
                "valid_token_mask must be contiguous CUDA");
    TORCH_CHECK(valid_token_mask.scalar_type() == at::ScalarType::Bool,
                "valid_token_mask must be bool");
    TORCH_CHECK(valid_token_mask.numel() == rows64,
                "mask must contain one value per token");
    TORCH_CHECK(valid_token_mask.device() == residual.device(), "mask device mismatch");
  }

  c10::cuda::CUDAGuard device_guard(residual.device());
  auto residual_out = torch::empty_like(residual);
  auto norm_out = torch::empty_like(residual);
  if (rows64 == 0) return {residual_out, norm_out};

  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  residual_bias_layer_norm_512_kernel<<<static_cast<int>(rows64), kNormThreads, 0, stream>>>(
      residual.data_ptr<float>(), delta.data_ptr<float>(), bias.data_ptr<float>(),
      weight.data_ptr<float>(), norm_bias.data_ptr<float>(),
      has_mask ? valid_token_mask.data_ptr<bool>() : nullptr,
      residual_out.data_ptr<float>(), norm_out.data_ptr<float>(),
      static_cast<float>(eps), zero_invalid_norm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual_out, norm_out};
}
