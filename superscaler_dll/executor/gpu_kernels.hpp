#pragma once

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

#define BLOCK 512

inline dim3 cuda_gridsize_1d(int n){
    int x = (n-1) / BLOCK + 1;
    dim3 d = {(uint) x, 1, 1};
    return d;
}

template <class T>
void SumKernelGPU(const T* buffer, T* memory, size_t offset, size_t num_elements);

struct SumKernelGPUImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t offset, size_t num_elements) {
        SumKernelGPU<T>(buffer, memory, offset, num_elements);
    }
};

#endif
