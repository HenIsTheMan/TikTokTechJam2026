#include <torch/extension.h>

#include <vector>

torch::Tensor bias_gelu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& bias);

std::vector<torch::Tensor> residual_bias_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& delta,
    const torch::Tensor& bias,
    const torch::Tensor& weight,
    const torch::Tensor& layer_norm_bias,
    const torch::Tensor& valid_token_mask,
    double eps,
    bool zero_invalid_norm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("bias_gelu", &bias_gelu_cuda,
             "Fused FP32 bias + exact GELU (CUDA)");
  module.def("residual_bias_layer_norm", &residual_bias_layer_norm_cuda,
             "Fused FP32 residual + bias + mask + LayerNorm (CUDA)");
}
