// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <gtest/gtest.h>

#include "scale_task.hpp"

TEST(ScaleTask, FloatScaleTask)
{
    float scale = 0.5;
    size_t num_element = 8;

    std::vector<float> memory(num_element);
    std::vector<float> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    ScaleTask<float, ScaleKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, ScaleKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, FloatDivTask)
{
    float scale = 2.0;
    size_t num_element = 8;

    std::vector<float> memory(num_element);
    std::vector<float> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    ScaleTask<float, DivKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, DivKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, DoubleScaleTask)
{
    double scale = 0.5;
    size_t num_element = 8;

    std::vector<double> memory(num_element);
    std::vector<double> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    ScaleTask<double, ScaleKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, ScaleKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, DoubleDivTask)
{
    double scale = 2.0;
    size_t num_element = 8;

    std::vector<double> memory(num_element);
    std::vector<double> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    ScaleTask<double, DivKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, DivKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, IntScaleTask)
{
    int scale = 2;
    size_t num_element = 8;

    std::vector<int> memory(num_element);
    std::vector<int> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 4;
        reference[i] = 8;
    }

    ScaleTask<int, ScaleKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, ScaleKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_EQ(memory[i] ,reference[i]);
}

TEST(ScaleTask, IntDivTask)
{
    int scale = 2;
    size_t num_element = 8;

    std::vector<int> memory(num_element);
    std::vector<int> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 4;
        reference[i] = 2;
    }

    ScaleTask<int, DivKernelCPUImpl> scale_task(nullptr, nullptr, &memory[0], scale, DivKernelCPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < num_element; ++i)
        ASSERT_EQ(memory[i] ,reference[i]);
}

#ifdef GPU_SWITCH
TEST(ScaleTask, FloatScaleTaskGPU)
{
    float scale = 0.5;
    size_t num_element = 8;

    std::vector<float> memory(num_element);
    std::vector<float> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    float *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(float));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(float), cudaMemcpyHostToDevice);

    ScaleTask<float, ScaleKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, ScaleKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, FloatDivTaskGPU)
{
    float scale = 2.0;
    size_t num_element = 8;

    std::vector<float> memory(num_element);
    std::vector<float> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    float *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(float));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(float), cudaMemcpyHostToDevice);

    ScaleTask<float, DivKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, DivKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}
TEST(ScaleTask, DoubleScaleTaskGPU)
{
    double scale = 0.5;
    size_t num_element = 8;

    std::vector<double> memory(num_element);
    std::vector<double> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    double *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(double));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(double), cudaMemcpyHostToDevice);

    ScaleTask<double, ScaleKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, ScaleKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}

TEST(ScaleTask, DoubleDivTaskGPU)
{
    double scale = 2.0;
    size_t num_element = 8;

    std::vector<double> memory(num_element);
    std::vector<double> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 1.0;
        reference[i] = 0.5;
    }

    double *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(double));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(double), cudaMemcpyHostToDevice);

    ScaleTask<double, DivKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, DivKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_LT(abs(memory[i] - reference[i]),  1e-6);
}
TEST(ScaleTask, IntScaleTaskGPU)
{
    int scale = 2;
    size_t num_element = 8;

    std::vector<int> memory(num_element);
    std::vector<int> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 4;
        reference[i] = 8;
    }

    int *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(int));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(int), cudaMemcpyHostToDevice);

    ScaleTask<int, ScaleKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, ScaleKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(int), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_EQ(memory[i] ,reference[i]);
}

TEST(ScaleTask, IntDivTaskGPU)
{
    int scale = 2;
    size_t num_element = 8;

    std::vector<int> memory(num_element);
    std::vector<int> reference(num_element);
    for(size_t i = 0; i < num_element; ++i) {
        memory[i] = 4;
        reference[i] = 2;
    }

    int *memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&memory_gpu, num_element * sizeof(int));
    cudaMemcpy(memory_gpu, memory.data(), num_element * sizeof(int), cudaMemcpyHostToDevice);

    ScaleTask<int, DivKernelGPUImpl> scale_task(nullptr, nullptr, &memory_gpu[0], scale, DivKernelGPUImpl(), num_element);
    scale_task();
    ASSERT_EQ(scale_task.get_state(), TaskState::e_success);

    cudaMemcpy(&memory[0], memory_gpu, num_element * sizeof(int), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < num_element; ++i)
        ASSERT_EQ(memory[i] ,reference[i]);
}
#endif