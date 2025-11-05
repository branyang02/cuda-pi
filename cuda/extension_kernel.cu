#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void add_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                           const scalar_t value, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + value;
    }
}

torch::Tensor cuda_add_forward(torch::Tensor input, float value) {
    auto output = torch::zeros_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "add_forward_cuda", ([&] {
                                   add_kernel<scalar_t><<<blocks, threads>>>(
                                       input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                       static_cast<scalar_t>(value), size);
                               }));

    return output;
}
