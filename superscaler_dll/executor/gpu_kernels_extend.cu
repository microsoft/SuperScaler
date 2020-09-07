#include "gpu_kernels.cu"

template void SumKernelGPU(const float* buffer, float* memory, size_t offset, size_t num_elements);
template void SumKernelGPU(const double* buffer, double* memory, size_t offset, size_t num_elements);
template void SumKernelGPU(const int* buffer, int* memory, size_t offset, size_t num_elements);
