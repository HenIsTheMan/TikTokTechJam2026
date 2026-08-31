#include <ATen/ATen.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/extension.h>

#include <vector>

torch::Tensor transformer_forward_sdpa_cuda(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& parameters,
    const torch::Tensor& final_norm_weight,
    const torch::Tensor& final_norm_bias,
    double eps,
    int64_t num_heads,
    bool causal) {
  constexpr int64_t kParamsPerLayer = 12;
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 3, "input must have shape [B, S, D]");
  TORCH_CHECK(parameters.size() % kParamsPerLayer == 0,
              "invalid packed parameter count");
  const int64_t layers = parameters.size() / kParamsPerLayer;
  const int64_t batch = input.size(0);
  const int64_t sequence = input.size(1);
  const int64_t d_model = input.size(2);
  TORCH_CHECK(d_model % num_heads == 0, "invalid head count");
  const int64_t head_dim = d_model / num_heads;
  auto residual = input;

  for (int64_t layer = 0; layer < layers; ++layer) {
    const int64_t p = layer * kParamsPerLayer;
    // qkv weight/bias, output weight/bias, norm1 weight/bias,
    // norm2 weight/bias, FFN-in weight/bias, FFN-out weight/bias.
    auto norm1 = at::layer_norm(
        residual, {d_model}, parameters[p + 4], parameters[p + 5], eps, true);
    auto qkv = at::linear(norm1, parameters[p], parameters[p + 1]);
    qkv = qkv.view({batch, sequence, 3, num_heads, head_dim})
              .permute({2, 0, 3, 1, 4});
    auto query = qkv.select(0, 0);
    auto key = qkv.select(0, 1);
    auto value = qkv.select(0, 2);
    auto context = at::scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, causal, std::nullopt, false);
    context = context.transpose(1, 2).contiguous().view(
        {batch, sequence, d_model});
    residual = residual + at::linear(
        context, parameters[p + 2], parameters[p + 3]);
    auto norm2 = at::layer_norm(
        residual, {d_model}, parameters[p + 6], parameters[p + 7], eps, true);
    auto hidden = at::gelu(
        at::linear(norm2, parameters[p + 8], parameters[p + 9]), "none");
    residual = residual + at::linear(
        hidden, parameters[p + 10], parameters[p + 11]);
  }
  return at::layer_norm(
      residual, {d_model}, final_norm_weight, final_norm_bias, eps, true);
}

torch::Tensor layer_norm_half_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps);

torch::Tensor bias_gelu_cuda(const torch::Tensor& input, const torch::Tensor& bias);
torch::Tensor bias_gelu_half_cuda(
    const torch::Tensor& input, const torch::Tensor& bias);

std::vector<torch::Tensor> mixed_residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool norm_output_half,
    bool zero_invalid_norm);

torch::Tensor transformer_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& initial_norm_weight,
    const torch::Tensor& initial_norm_bias,
    const std::vector<torch::Tensor>& parameters,
    const torch::Tensor& empty_mask,
    double eps,
    int64_t num_heads) {
  constexpr int64_t kLayers = 6;
  constexpr int64_t kParamsPerLayer = 15;
  constexpr int64_t kDModel = 512;
  TORCH_CHECK(parameters.size() == kLayers * kParamsPerLayer,
              "expected 90 packed layer tensors");
  TORCH_CHECK(input.dim() == 3 && input.size(2) == kDModel,
              "fast Transformer path requires [B, S, 512]");
  TORCH_CHECK(kDModel % num_heads == 0, "invalid head count");

  const int64_t batch = input.size(0);
  const int64_t sequence = input.size(1);
  const int64_t head_dim = kDModel / num_heads;
  auto norm = layer_norm_half_cuda(
      input, initial_norm_weight, initial_norm_bias, eps);
  auto residual = input;

  for (int64_t layer = 0; layer < kLayers; ++layer) {
    const int64_t p = layer * kParamsPerLayer;
    // [QKV weight, QKV bias, attention output weight, output bias,
    //  norm2 weight/bias, FFN-in float weight/bias, FFN-in half weight/bias,
    //  FFN-out float/half weight, FFN-out bias, next norm weight/bias]
    auto qkv = at::linear(norm, parameters[p], parameters[p + 1]);
    qkv = qkv.view({batch, sequence, 3, num_heads, head_dim})
              .permute({2, 0, 3, 1, 4});
    auto query = qkv.select(0, 0);
    auto key = qkv.select(0, 1);
    auto value = qkv.select(0, 2);
    auto context = at::scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, false, std::nullopt, false);
    context = context.transpose(1, 2).contiguous().view(
        {batch, sequence, kDModel});
    auto attention_delta = at::linear(context, parameters[p + 2], std::nullopt);
    auto attention_outputs = mixed_residual_bias_layer_norm_cuda(
        residual, attention_delta, parameters[p + 3], parameters[p + 4],
        parameters[p + 5], empty_mask, eps, false, false);
    residual = attention_outputs[0];
    auto norm2 = attention_outputs[1];

    torch::Tensor hidden;
    if (layer == kLayers - 1) {
      hidden = at::gelu(
          at::linear(norm2.to(at::ScalarType::Half), parameters[p + 8],
                     parameters[p + 9]),
          "none");
    } else {
      auto ffn_input = at::linear(norm2, parameters[p + 6], std::nullopt);
      hidden = layer == 0
          ? bias_gelu_cuda(ffn_input, parameters[p + 7])
          : bias_gelu_half_cuda(ffn_input, parameters[p + 7]);
    }

    auto ffn_delta = layer == 0
        ? at::linear(hidden, parameters[p + 10], std::nullopt)
        : at::linear(hidden, parameters[p + 11], std::nullopt);
    auto layer_outputs = mixed_residual_bias_layer_norm_cuda(
        residual, ffn_delta, parameters[p + 12], parameters[p + 13],
        parameters[p + 14], empty_mask, eps, layer + 1 < kLayers,
        layer + 1 == kLayers);
    residual = layer_outputs[0];
    norm = layer_outputs[1];
  }
  return norm;
}
