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

