#include <array>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <type_traits>
#include "gtest/gtest.h"

#include "session.hpp"

template <typename DataType>
void display_buffer_content(const DataType* ptr, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%lf ", ptr[i]);
    }
    printf("\n");
}

template <typename DataType>
struct AllReduceOp
{
    DataType* cpu_dbuff;
    const DataType* cpu_expected;
    DataType* gpu_dbuff;
    size_t size;
    std::string name;
};

template <typename DataType>
AllReduceOp<DataType> make_allreduce_op(std::string name, size_t size)
{
    return {new DataType[size], nullptr, nullptr, size, name};
}

template <typename DataType>
DataType* fill_allreduce_cpu_tensor(AllReduceOp<DataType>& op1, AllReduceOp<DataType>& op2)
{
    assert(op1.name == op2.name);
    assert(op1.size == op2.size);
    DataType* expected = new DataType[op1.size];
    for (int i = 0; i < op1.size; i++)
    {
        (op1.cpu_dbuff)[i] = rand() * 1.0;
        (op2.cpu_dbuff)[i] = rand() * 1.0;
        expected[i] = ((op1.cpu_dbuff)[i] + (op2.cpu_dbuff)[i]) / 2;
    }
    op1.cpu_expected = op2.cpu_expected = static_cast<const DataType*>(expected);
    return expected;
}

template <typename DataType>
void assert_values(DataType* cpu_src, const DataType* cpu_expected, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        ASSERT_EQ(cpu_src[i], cpu_expected[i]) << "Vectors x and y differ at index " << i;
}

template <>
void assert_values(float* cpu_src, const float* cpu_expected, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        ASSERT_FLOAT_EQ(cpu_src[i], cpu_expected[i]) << "Vectors x and y differ at index " << i;
}

template <typename DataType, typename Plan>
void process_func(std::vector<AllReduceOp<DataType>> graph, Plan plan)
{
    superscaler::Session sess;
    if (std::is_same<Plan, std::string>::value)
        sess.Create(plan.c_str());
    else
        sess.Create(plan);

    // std::string planPath = std::to_string(grank) + ".json";
    printf("-- [Host: %d Device: %d] is running\n", sess.GetHostId(), sess.GetDeviceId());

    cudaSetDevice(sess.GetDeviceId());
    for (auto& op : graph)
    {
        DataType* gpu_buff = NULL;
        cudaMalloc((void**)(&gpu_buff), op.size * sizeof(DataType));
        cudaMemset(gpu_buff, 0, op.size * sizeof(DataType));
        cudaMemcpy(gpu_buff, op.cpu_dbuff, op.size * sizeof(DataType), cudaMemcpyHostToDevice);
        op.gpu_dbuff = gpu_buff;
    }

    for (auto& op : graph)
    {
#ifndef NDEBUG
        printf("-- Before allReduce: \n");
        printf("\noriginal: ");
        display_buffer_content(op.cpu_dbuff, 8);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
        sess.AllReduce(op.name.c_str(), op.gpu_dbuff, op.size, nullptr);

#ifndef NDEBUG
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        printf("-- After allReduce: ");
        printf("-- Avg time per allreduce: %lf [s]\n",
               1.0 * std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                   1000000);
#endif
    }

    for (auto& op : graph)
    {
        cudaMemcpy(op.cpu_dbuff, op.gpu_dbuff, op.size * sizeof(DataType), cudaMemcpyDeviceToHost);
        assert_values(op.cpu_dbuff, op.cpu_expected, op.size);
#ifndef NDEBUG
        display_buffer_content(op.cpu_dbuff, 8);
#endif
    }

    cudaSetDevice(sess.GetDeviceId());
    for (auto& op : graph)
    {
        cudaFree(op.gpu_dbuff);
    }

    sess.Close();

    printf("-- Test Done!\n");
}

std::unordered_map<std::string, size_t> parse_json(superscaler::util::json j)
{
    std::unordered_map<std::string, size_t> ops;

    for (auto element : j["tasks"])
    {
        auto tensor_name = element["tensor_name"];
        auto tensor_shape = element["output_shapes"];
        size_t len = 1;
        for (auto i : tensor_shape[0].get<std::vector<int>>())
            len *= i;
        if (ops.find(tensor_name) == ops.end())
        {
            VLOG(1)<< tensor_name << ": " << len;
            ops[tensor_name] = (len);
        }
    }
    return ops;
}

TEST(TwoPeerGraphAllReduceTest, PCIE_RING_TWO_WORKERS)
{
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));

    std::string plan0 = dir_path + "/plan/0/plan.json";
    std::string plan1 = dir_path + "/plan/1/plan.json";

    auto plan = superscaler::util::JsonParser::load_from(plan0);
    auto ops = parse_json(plan);

    std::vector<AllReduceOp<float>> graph0;
    std::vector<AllReduceOp<float>> graph1;

    std::vector<float*> op_expected;

    for (auto& op : ops)
    {
        graph0.push_back(make_allreduce_op<float>(op.first, op.second));
        graph1.push_back(make_allreduce_op<float>(op.first, op.second));
    }

    srand(time(0));
    for (int i = 0; i < graph0.size(); i++)
    {
        op_expected.push_back(fill_allreduce_cpu_tensor(graph0[i], graph1[i]));
    }

    pid_t pid = fork();
    int status;
    if (pid == 0)
    {
        process_func(graph0, plan0);
    }
    else
    {
        process_func(graph1, plan1);
        wait(&status);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
