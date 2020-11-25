// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <gtest/gtest.h>

#include "reduction_task.hpp"

TEST(ReductionTask, FloatSumTask)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.3;
    }

    size_t num_element = 8;

    ReductionTask<float, SumKernelCPUImpl> sum_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], SumKernelCPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}


TEST(ReductionTask, FloatCopyTask)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.1;
    }

    size_t num_element = 8;

    ReductionTask<float, CopyKernelCPUImpl> copy_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], CopyKernelCPUImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, DoubleSumTask)
{
    std::vector<double> input_buffer(8);
    std::vector<double> output_memory(8);
    std::vector<double> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.3;
    }

    size_t num_element = 8;

    ReductionTask<double, SumKernelCPUImpl> sum_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], SumKernelCPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}


TEST(ReductionTask, DoubleCopyTask)
{
    std::vector<double> input_buffer(8);
    std::vector<double> output_memory(8);
    std::vector<double> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.1;
    }

    size_t num_element = 8;

    ReductionTask<double, CopyKernelCPUImpl> copy_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], CopyKernelCPUImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, UINT8SumTask)
{
    std::vector<uint8_t> input_buffer(8);
    std::vector<uint8_t> output_memory(8);
    std::vector<uint8_t> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 3;
    }

    size_t num_element = 8;

    ReductionTask<uint8_t, SumKernelCPUImpl> sum_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], SumKernelCPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}


TEST(ReductionTask, UINT8CopyTask)
{
    std::vector<uint8_t> input_buffer(8);
    std::vector<uint8_t> output_memory(8);
    std::vector<uint8_t> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 1;
    }

    size_t num_element = 8;

    ReductionTask<uint8_t, CopyKernelCPUImpl> copy_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], CopyKernelCPUImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}

TEST(ReductionTask, INTSumTask)
{
    std::vector<int> input_buffer(8);
    std::vector<int> output_memory(8);
    std::vector<int> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 3;
    }

    size_t num_element = 8;

    ReductionTask<int, SumKernelCPUImpl> sum_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], SumKernelCPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}


TEST(ReductionTask, INTCopyTask)
{
    std::vector<int> input_buffer(8);
    std::vector<int> output_memory(8);
    std::vector<int> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 1;
    }

    size_t num_element = 8;

    ReductionTask<int, CopyKernelCPUImpl> copy_task(nullptr, nullptr, input_buffer.data(), &output_memory[0], CopyKernelCPUImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}

TEST(ReductionTask, FirstHalfTest)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.2;
    }

    size_t offset = 0;
    size_t num_element = 4;
    for(size_t i = offset; i < offset+num_element; ++i)
        reference[i] = 0.3;

    ReductionTask<float, SumKernelCPUImpl> sum_task(nullptr, nullptr, input_buffer.data() + offset, &output_memory[0] + offset, SumKernelCPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);
    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

#ifdef GPU_SWITCH
TEST(ReductionTask, FloatSumTaskGPU)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.3;
    }

    float *input_buffer_gpu = nullptr;
    float *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(float));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(float));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<float, SumKernelGPUImpl> sum_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SumKernelGPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, FloatCopyTaskGPU)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.1;
    }

    float *input_buffer_gpu = nullptr;
    float *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(float));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(float));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<float, SynchronizedCopyKernelImpl> copy_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SynchronizedCopyKernelImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, DoubleSumTaskGPU)
{
    std::vector<double> input_buffer(8);
    std::vector<double> output_memory(8);
    std::vector<double> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.3;
    }

    double *input_buffer_gpu = nullptr;
    double *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(double));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(double));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(double), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<double, SumKernelGPUImpl> sum_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SumKernelGPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, DoubleCopyTaskGPU)
{
    std::vector<double> input_buffer(8);
    std::vector<double> output_memory(8);
    std::vector<double> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.1;
    }

    double *input_buffer_gpu = nullptr;
    double *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(double));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(double));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(double), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<double, SynchronizedCopyKernelImpl> copy_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SynchronizedCopyKernelImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
}

TEST(ReductionTask, IntSumTaskGPU)
{
    std::vector<int> input_buffer(8);
    std::vector<int> output_memory(8);
    std::vector<int> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 3;
    }

    int *input_buffer_gpu = nullptr;
    int *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(int));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(int));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(int), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<int, SumKernelGPUImpl> sum_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SumKernelGPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}

TEST(ReductionTask, IntCopyTaskGPU)
{
    std::vector<int> input_buffer(8);
    std::vector<int> output_memory(8);
    std::vector<int> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 1;
        output_memory[i] = 2;
        reference[i] = 1;
    }

    int *input_buffer_gpu = nullptr;
    int *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(int));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(int));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(int), cudaMemcpyHostToDevice);

    size_t num_element = 8;

    ReductionTask<int, SynchronizedCopyKernelImpl> copy_task(nullptr, nullptr, input_buffer_gpu, output_memory_gpu, SynchronizedCopyKernelImpl(), num_element);
    copy_task();
    ASSERT_EQ(copy_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i)
        ASSERT_EQ(output_memory[i] , reference[i]);
}

TEST(ReductionTask, FirstHalfTestGPU)
{
    std::vector<float> input_buffer(8);
    std::vector<float> output_memory(8);
    std::vector<float> reference(8);
    for(size_t i = 0; i < 8; ++i) {
        input_buffer[i] = 0.1;
        output_memory[i] = 0.2;
        reference[i] = 0.2;
    }

    float *input_buffer_gpu = nullptr;
    float *output_memory_gpu = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&input_buffer_gpu, 8 * sizeof(float));
    cudaMemcpy(input_buffer_gpu, input_buffer.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&output_memory_gpu, 8 * sizeof(float));
    cudaMemcpy(output_memory_gpu, output_memory.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

    
    size_t offset = 0;
    size_t num_element = 4;
    for(size_t i = offset; i < offset+num_element; ++i)
        reference[i] = 0.3;

    ReductionTask<float, SumKernelGPUImpl> sum_task(nullptr, nullptr, input_buffer_gpu + offset, output_memory_gpu + offset, SumKernelGPUImpl(), num_element);
    sum_task();
    ASSERT_EQ(sum_task.get_state(), TaskState::e_success);

    cudaMemcpy(&output_memory[0], output_memory_gpu, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 8; ++i){
        ASSERT_LT(abs(output_memory[i] - reference[i]),  1e-6);
    }
}
#endif