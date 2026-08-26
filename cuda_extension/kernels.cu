#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kModelWidth = 512;
constexpr int kFfnWidth = 2048;
constexpr int kNormThreads = 128;
constexpr int kElementwiseThreads = 256;
constexpr float kInvSqrtTwo = 0.70710678118654752440f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// One block owns one token row. With width 512, every thread processes one
// float4. This shape-specific path avoids intermediate residual, bias and mask
// tensors and performs both LayerNorm outputs in one global-memory pass.
__global__ __launch_bounds__(kNormThreads)
void residual_bias_layer_norm_512_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ delta,
    const float* __restrict__ bias,
    const float* __restrict__ weight,
    const float* __restrict__ layer_norm_bias,
    const bool* __restrict__ valid_token_mask,
    float* __restrict__ residual_output,
    float* __restrict__ norm_output,
    float eps,
    bool zero_invalid_norm) {
  const int row = blockIdx.x;
  const bool valid = valid_token_mask == nullptr || valid_token_mask[row];
  const int vector_index = row * (kModelWidth / 4) + threadIdx.x;

  float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (valid) {
    const float4 residual4 = reinterpret_cast<const float4*>(residual)[vector_index];
    const float4 delta4 = reinterpret_cast<const float4*>(delta)[vector_index];
    const float4 bias4 = reinterpret_cast<const float4*>(bias)[threadIdx.x];
    value.x = residual4.x + delta4.x + bias4.x;
    value.y = residual4.y + delta4.y + bias4.y;
    value.z = residual4.z + delta4.z + bias4.z;
    value.w = residual4.w + delta4.w + bias4.w;
  }
  reinterpret_cast<float4*>(residual_output)[vector_index] = value;

  float sum = value.x + value.y + value.z + value.w;
  float square_sum = value.x * value.x + value.y * value.y +
                     value.z * value.z + value.w * value.w;
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);

  __shared__ float warp_sums[4];
  __shared__ float warp_square_sums[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
    warp_square_sums[warp] = square_sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = lane < 4 ? warp_sums[lane] : 0.0f;
    square_sum = lane < 4 ? warp_square_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
      warp_sums[0] = sum;
      warp_square_sums[0] = square_sum;
    }
  }
  __syncthreads();

  const float mean = warp_sums[0] * (1.0f / kModelWidth);
  // Match LayerNorm's population variance. Clamp protects rsqrt from a tiny
  // negative caused by sum/square-sum rounding.
  const float variance = fmaxf(
      warp_square_sums[0] * (1.0f / kModelWidth) - mean * mean, 0.0f);
  const float inverse_std = rsqrtf(variance + eps);
  const float4 weight4 = reinterpret_cast<const float4*>(weight)[threadIdx.x];
  const float4 norm_bias4 =
      reinterpret_cast<const float4*>(layer_norm_bias)[threadIdx.x];
  float4 normalized;
  normalized.x = (value.x - mean) * inverse_std * weight4.x + norm_bias4.x;
  normalized.y = (value.y - mean) * inverse_std * weight4.y + norm_bias4.y;
  normalized.z = (value.z - mean) * inverse_std * weight4.z + norm_bias4.z;
  normalized.w = (value.w - mean) * inverse_std * weight4.w + norm_bias4.w;
  if (!valid && zero_invalid_norm) {
    normalized = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  reinterpret_cast<float4*>(norm_output)[vector_index] = normalized;
}

__global__ void bias_gelu_2048_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t vector_elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vector_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int bias_index = static_cast<int>(index % (kFfnWidth / 4));
    const float4 input4 = reinterpret_cast<const float4*>(input)[index];
    const float4 bias4 = reinterpret_cast<const float4*>(bias)[bias_index];
    float4 result;
    const float x0 = input4.x + bias4.x;
    const float x1 = input4.y + bias4.y;
    const float x2 = input4.z + bias4.z;
    const float x3 = input4.w + bias4.w;
    result.x = 0.5f * x0 * (1.0f + erff(x0 * kInvSqrtTwo));
    result.y = 0.5f * x1 * (1.0f + erff(x1 * kInvSqrtTwo));
    result.z = 0.5f * x2 * (1.0f + erff(x2 * kInvSqrtTwo));
    result.w = 0.5f * x3 * (1.0f + erff(x3 * kInvSqrtTwo));
    reinterpret_cast<float4*>(output)[index] = result;
  }
}

void check_cuda_fp32_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float,
              name, " must have float32 dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor bias_gelu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& bias) {
  check_cuda_fp32_contiguous(input, "input");
  check_cuda_fp32_contiguous(bias, "bias");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == kFfnWidth,
              "bias_gelu requires last dimension 2048");
  TORCH_CHECK(bias.dim() == 1 && bias.numel() == kFfnWidth,
              "bias must contain 2048 elements");
  TORCH_CHECK(input.device() == bias.device(), "device mismatch");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t vector_elements = input.numel() / 4;
  if (vector_elements == 0) return output;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (vector_elements + kElementwiseThreads - 1) / kElementwiseThreads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  bias_gelu_2048_kernel<<<blocks, kElementwiseThreads, 0, stream>>>(
      input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
      vector_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& layer_norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool zero_invalid_norm) {
  check_cuda_fp32_contiguous(residual, "residual");
  check_cuda_fp32_contiguous(delta, "delta");
  check_cuda_fp32_contiguous(bias, "bias");
  check_cuda_fp32_contiguous(weight, "weight");
  check_cuda_fp32_contiguous(layer_norm_bias, "layer_norm_bias");
  TORCH_CHECK(residual.sizes() == delta.sizes(), "residual/delta shape mismatch");
  TORCH_CHECK(residual.dim() == 3 && residual.size(-1) == kModelWidth,
              "residual_bias_layer_norm requires [B, S, 512] tensors");
  TORCH_CHECK(bias.numel() == kModelWidth && weight.numel() == kModelWidth &&
                  layer_norm_bias.numel() == kModelWidth,
              "bias and LayerNorm parameters must contain 512 elements");
  TORCH_CHECK(residual.device() == delta.device() && residual.device() == bias.device() &&
                  residual.device() == weight.device() &&
                  residual.device() == layer_norm_bias.device(),
              "device mismatch");

  const int64_t rows = residual.numel() / kModelWidth;
  const bool has_mask = valid_token_mask.numel() != 0;
  if (has_mask) {
    TORCH_CHECK(valid_token_mask.is_cuda() && valid_token_mask.is_contiguous(),
                "valid_token_mask must be contiguous CUDA memory");
    TORCH_CHECK(valid_token_mask.scalar_type() == at::ScalarType::Bool,
                "valid_token_mask must have bool dtype");
    TORCH_CHECK(valid_token_mask.numel() == rows,
                "valid_token_mask must contain one value per token");
    TORCH_CHECK(valid_token_mask.device() == residual.device(), "mask device mismatch");
  }

  c10::cuda::CUDAGuard device_guard(residual.device());
  auto residual_output = torch::empty_like(residual);
  auto norm_output = torch::empty_like(residual);
  if (rows == 0) return {residual_output, norm_output};
  const bool* mask = has_mask ? valid_token_mask.data_ptr<bool>() : nullptr;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  residual_bias_layer_norm_512_kernel<<<rows, kNormThreads, 0, stream>>>(
      residual.data_ptr<float>(), delta.data_ptr<float>(), bias.data_ptr<float>(),
      weight.data_ptr<float>(), layer_norm_bias.data_ptr<float>(), mask,
      residual_output.data_ptr<float>(), norm_output.data_ptr<float>(),
      static_cast<float>(eps), zero_invalid_norm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual_output, norm_output};
}
