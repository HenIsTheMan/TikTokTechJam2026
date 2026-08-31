#include <torch/extension.h>

#include <vector>

torch::Tensor bias_gelu_cuda(const torch::Tensor& input, const torch::Tensor& bias);

torch::Tensor bias_gelu_half_cuda(
    const torch::Tensor& input,
    const torch::Tensor& bias);

torch::Tensor layer_norm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps);

torch::Tensor layer_norm_mask_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    double eps);

torch::Tensor layer_norm_half_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps);

std::vector<torch::Tensor> residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool zero_invalid_norm);

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
    int64_t num_heads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("bias_gelu", &bias_gelu_cuda, "Fused FP32 bias + exact GELU (CUDA)");
  module.def("bias_gelu_half", &bias_gelu_half_cuda,
             "Fused FP32 bias + exact GELU with FP16 output (CUDA)");
  module.def("layer_norm", &layer_norm_cuda,
             "Shape-specialized FP32 LayerNorm (CUDA)");
  module.def("layer_norm_mask", &layer_norm_mask_cuda,
             "Shape-specialized FP32 LayerNorm + token mask (CUDA)");
  module.def("layer_norm_half", &layer_norm_half_cuda,
             "Shape-specialized FP32 LayerNorm with FP16 output (CUDA)");
  module.def("residual_bias_layer_norm", &residual_bias_layer_norm_cuda,
             "Fused FP32 residual + bias + LayerNorm (CUDA)");
  module.def("mixed_residual_bias_layer_norm",
             &mixed_residual_bias_layer_norm_cuda,
             "Fused mixed-precision residual + bias + LayerNorm (CUDA)");
  module.def("transformer_forward", &transformer_forward_cuda,
             "Six-layer mixed-precision Transformer forward (CUDA)");
}
