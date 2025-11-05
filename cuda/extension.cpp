#include <torch/extension.h>

#include <vector>

// Forward declarations of CUDA functions
torch::Tensor cuda_add_forward(torch::Tensor input, float value);

// C++ interface
torch::Tensor add_forward(torch::Tensor input, float value) {
    return cuda_add_forward(input, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_forward", &add_forward, "Add a value to tensor (CUDA)");
}
