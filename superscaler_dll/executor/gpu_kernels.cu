#include "gpu_kernels.hpp"

template <class T>
__global__ static void SumKernel(const T* buffer, T* memory, size_t offset, size_t num_elements)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= offset+num_elements || index < offset) return;
    memory[index] += buffer[index];
}

template <class T>
void SumKernelGPU(const T* buffer, T* memory, size_t offset, size_t num_elements) {
    SumKernel<T><<<cuda_gridsize_1d(offset+num_elements), BLOCK, 0, 0>>>(buffer, memory, offset, num_elements);
}
