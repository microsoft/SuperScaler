#include <array>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "gtest/gtest.h"

#include "session.hpp"

template <typename DataType>
struct TestProcessContext
{
    DataType* ioput;
    DataType* expected;
    size_t size;
    superscaler::util::json plan;
};

template <typename DataType>
void display_buffer_content(DataType* ptr, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%lf ", ptr[i]);
    }
    printf("\n");
}

template <typename DataType>
void assert_values(DataType* src, DataType* expected, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        ASSERT_EQ(src[i], expected[i]) << "Vectors x and y differ at index " << i;
}

template <>
void assert_values(float* src, float* expected, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        ASSERT_FLOAT_EQ(src[i], expected[i]) << "Vectors x and y differ at index " << i;
}

template <typename DataType>
void process_func(TestProcessContext<DataType> ctx)
{
    superscaler::Session sess;

    // sess.Create(planPath.c_str());
    sess.Create(ctx.plan);

    // std::string planPath = std::to_string(grank) + ".json";
    printf("-- [Host: %d Device: %d] is running\n", sess.GetHostId(), sess.GetDeviceId());

    DataType* sendbuff = NULL;
    cudaSetDevice(sess.GetDeviceId());
    cudaMalloc((void**)(&sendbuff), ctx.size * sizeof(DataType));
    cudaMemset(sendbuff, 0, ctx.size * sizeof(DataType));
    cudaMemcpy(sendbuff, ctx.ioput, ctx.size * sizeof(DataType), cudaMemcpyHostToDevice);

#ifndef NDEBUG
    printf("-- Before allReduce: ");
    display_buffer_content(ctx.ioput, 8);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    int steps = 10000;
    for (int i = 0; i < steps; ++i)
    {
        sess.AllReduce("AllReduce_0", sendbuff, ctx.size, nullptr);
    }

#ifndef NDEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    printf("-- After allReduce: ");
    printf("-- Avg time per allreduce: %lf [s]\n",
           1.0 * std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
               1000000 / steps);
#endif

    cudaMemcpy(ctx.ioput, sendbuff, ctx.size * sizeof(DataType), cudaMemcpyDeviceToHost);

    assert_values(ctx.ioput, ctx.expected, ctx.size);

#ifndef NDEBUG
    display_buffer_content(ctx.ioput, 8);
#endif

    cudaSetDevice(sess.GetDeviceId());
    cudaFree(sendbuff);
    sess.Close();
    printf("-- Test Done!\n");
}

TEST(TwoPeerSingleAllReduceTest, PCIE_RING)
{
    auto plan0 = R"(
{
    "host_id": "1",
    "device_id": "0",
    "device_type": "GPU",
    "num_peers": "2",
    "recv_buffer_size": 61440,
    "tasks": [
        {
            "index": 0,
            "input_ids": [],
            "key": "rendez_0",
            "offset": 0,
            "op": "Send",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "",
            "related_id": 5,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "1",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 1,
            "input_ids": [
                0
            ],
            "key": "rendez_1",
            "offset": 15360,
            "op": "Recv",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "sum",
            "related_id": 4,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "1",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 2,
            "input_ids": [
                1
            ],
            "key": "rendez_2",
            "offset": 15360,
            "op": "Send",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "",
            "related_id": 7,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "1",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 3,
            "input_ids": [
                2
            ],
            "key": "rendez_3",
            "offset": 0,
            "op": "Recv",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "copy",
            "related_id": 6,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "1",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        }
    ]
}
)"_json;

    auto plan1 = R"(
{
    "host_id": "1",
    "device_id": "1",
    "device_type": "GPU",
    "num_peers": "2",
    "recv_buffer_size": 61440,
    "tasks": [
        {
            "index": 4,
            "input_ids": [],
            "key": "rendez_1",
            "offset": 15360,
            "op": "Send",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "",
            "related_id": 1,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "0",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 5,
            "input_ids": [
                4
            ],
            "key": "rendez_0",
            "offset": 0,
            "op": "Recv",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "sum",
            "related_id": 0,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "0",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 6,
            "input_ids": [
                5
            ],
            "key": "rendez_3",
            "offset": 0,
            "op": "Send",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "",
            "related_id": 3,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "0",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        },
        {
            "index": 7,
            "input_ids": [
                6
            ],
            "key": "rendez_2",
            "offset": 15360,
            "op": "Recv",
            "output_shapes": [
                [
                    3072,
                    10
                ]
            ],
            "parent": "",
            "reduction": "copy",
            "related_id": 2,
            "route_index": 0,
            "route_type": "PCIE",
            "size": 15360,
            "target_device_id": "0",
            "target_device_type": "GPU",
            "target_host_id": "1",
            "tensor_name": "AllReduce_0",
            "tensor_type": "DT_FLOAT"
        }
    ]
}
)"_json;

    const int test_size = 30 * 1024;

    float tensor_0[test_size];
    float tensor_1[test_size];
    float expected[test_size];

    srand(time(0));
    for (int i = 0; i < test_size; i++)
    {
        tensor_0[i] = rand() % 10000 * 0.33;
        tensor_1[i] = rand() % 10000 * 0.33;
        expected[i] = (tensor_0[i] + tensor_1[i]) / 2;
    }
    pid_t pid = fork();
    int status;
    if (pid == 0)
    {
        process_func(TestProcessContext<float>{tensor_0, expected, test_size, plan0});
    }
    else
    {
        process_func(TestProcessContext<float>{tensor_1, expected, test_size, plan1});
        wait(&status);
    }
}
