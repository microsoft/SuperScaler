// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifdef GPU_SWITCH

#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "gpu_kernels.hpp"

#define BLOCK 512

inline dim3 cuda_gridsize_1d(int n){
    int x = (n-1) / BLOCK + 1;
    dim3 d = {(uint) x, 1, 1};
    return d;
}

template <class T>
__global__ static void SumKernel(const T* buffer, T* memory, size_t num_elements)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= num_elements) return;
    memory[index] += buffer[index];
}

template <class T>
void SumKernelGPUImpl::operator()(const T* buffer, T* memory,
    size_t num_elements, cudaStream_t stream) {
    SumKernel<T><<<cuda_gridsize_1d(num_elements), BLOCK, 0, stream>>>(
        buffer, memory, num_elements);
}

template <class T>
void SynchronizedCopyKernelImpl::operator()(const T* buffer, T* memory, size_t num_elements, cudaStream_t) {
    cudaMemcpy(memory, buffer, num_elements * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <class T>
__global__ static void ScaleKernel(T* memory, T scale, size_t num_elements)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= num_elements) return;
    memory[index] = memory[index] * scale;
}

template <class T>
void ScaleKernelGPUImpl::operator()(T* memory, T scale, size_t num_elements,
    cudaStream_t stream) {
    ScaleKernel<T><<<cuda_gridsize_1d(num_elements), BLOCK, 0, stream>>>(
        memory, scale, num_elements);
}

template <class T>
__global__ static void DivKernel(T* memory, T scale, size_t num_elements)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= num_elements) return;
    memory[index] = memory[index] / scale;
}

template <class T>
void DivKernelGPUImpl::operator()(T* memory, T scale, size_t num_elements,
    cudaStream_t stream) {
    DivKernel<T><<<cuda_gridsize_1d(num_elements), BLOCK, 0, stream>>>(
        memory, scale, num_elements);
}

template void SumKernelGPUImpl::operator()(const float* buffer, float* memory, size_t num_elements, cudaStream_t stream);
template void SumKernelGPUImpl::operator()(const double* buffer, double* memory, size_t num_elements, cudaStream_t stream);
template void SumKernelGPUImpl::operator()(const int* buffer, int* memory, size_t num_elements, cudaStream_t stream);
template void SynchronizedCopyKernelImpl::operator()(const float* buffer, float* memory, size_t num_elements, cudaStream_t);
template void SynchronizedCopyKernelImpl::operator()(const double* buffer, double* memory, size_t num_elements, cudaStream_t);
template void SynchronizedCopyKernelImpl::operator()(const int* buffer, int* memory, size_t num_elements, cudaStream_t);
template void ScaleKernelGPUImpl::operator()(float* memory, float scale, size_t num_elements, cudaStream_t stream);
template void ScaleKernelGPUImpl::operator()(double* memory, double scale, size_t num_elements, cudaStream_t stream);
template void ScaleKernelGPUImpl::operator()(int* memory, int scale, size_t num_elements, cudaStream_t stream);
template void DivKernelGPUImpl::operator()(float* memory, float scale, size_t num_elements, cudaStream_t stream);
template void DivKernelGPUImpl::operator()(double* memory, double scale, size_t num_elements, cudaStream_t stream);
template void DivKernelGPUImpl::operator()(int* memory, int scale, size_t num_elements, cudaStream_t stream);

#endif
