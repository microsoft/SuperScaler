#include <string>
#include <gtest/gtest.h>
#include <array>
#include <numeric>
#include <cstdint>

#include "shared_block.hpp"
#include "shared_pipe.hpp"
#include "utils.hpp"

TEST(CudaIPCSingle, SharedBlock)
{
    std::string input_data{ "1234567890" };
    char output_data[11];
    memset(output_data, 0, sizeof(output_data));
    SharedBlock block(input_data.size());
    auto cp_result = cudaMemcpy(block.get_buffer(), input_data.data(),
                                input_data.size(), cudaMemcpyDefault);
    ASSERT_EQ(cp_result, cudaSuccess);
    cp_result = cudaMemcpy(output_data, block.get_buffer(), input_data.size(),
                           cudaMemcpyDefault);
    ASSERT_EQ(cp_result, cudaSuccess);
    ASSERT_EQ(input_data, std::string{ output_data });
}

TEST(CudaIPCSingle, SharedTableSimpleCopy)
{
    std::string input_data{ "1234567890" };
    char output_data[11];
    memset(output_data, 0, sizeof(output_data));
    std::string test_name = get_thread_unique_name("SharedTableSimpleCopy");
    SharedTable<> table(test_name.c_str(), 1024);
    auto cp_result = cudaMemcpy(table.get_buffer(1), input_data.data(),
                                input_data.size(), cudaMemcpyDefault);
    ASSERT_EQ(cp_result, cudaSuccess);
    cp_result = cudaMemcpy(output_data, table.get_buffer(1), input_data.size(),
                           cudaMemcpyDefault);
    ASSERT_EQ(cp_result, cudaSuccess);
    ASSERT_EQ(input_data, std::string{ output_data });
    // ASSERT_SUCCESS;
}

TEST(CudaIPCSingle, SharedTableMultiGPU)
{
    std::string table_name = get_thread_unique_name("SharedTableSingleProcess");
    const int block_size = 1024;
    const char *target_str = "Hello World!";
    char buffer[block_size] = { 0 };
    {
        SharedTable<> shared_table1(table_name.c_str(), block_size);
        SharedTable<> shared_table2(table_name.c_str(), block_size);

        checkCudaErrors(cudaMemcpy(shared_table1.get_buffer(0), target_str,
                                   strlen(target_str), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(shared_table2.get_buffer(1),
                                   shared_table1.get_buffer(0),
                                   strlen(target_str), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(buffer, shared_table2.get_buffer(1),
                                   strlen(target_str), cudaMemcpyDefault));
    }
    ASSERT_TRUE(memcmp(buffer, target_str, strlen(target_str)) == 0);
}

TEST(CudaIPCSingle, SharedPipe)
{
    const size_t test_size = 1024;
    std::vector<uint8_t> input_data(test_size), output_data(test_size);
    srand(time(nullptr));
    for (auto &c : input_data) {
        c = random() % std::numeric_limits<uint8_t>::max();
    }
    std::string test_name = get_thread_unique_name("SharedPipe");
    SharedPipe shared_pipe(test_name.c_str(), 1, { 0 }, test_size + 1);
    size_t result;
    result = shared_pipe.write(input_data.data(), test_size, 0);
    ASSERT_EQ(result, test_size);
    result = shared_pipe.read(output_data.data(), test_size, 0);
    ASSERT_EQ(result, test_size);
    ASSERT_EQ(input_data, output_data);
}

TEST(CudaIPCSingle, DoubleSharedPipe)
{
    const size_t test_size = 1024;
    std::vector<uint8_t> input_data(test_size), output_data(test_size);
    srand(time(nullptr));
    for (auto &c : input_data) {
        c = random() % std::numeric_limits<uint8_t>::max();
    }
    std::string test_name1 = get_thread_unique_name("SharedPipe1");
    std::string test_name2 = get_thread_unique_name("SharedPipe2");
    SharedPipe shared_pipe1(test_name1.c_str(), 2, { 0, 1 }, test_size + 1);
    SharedPipe shared_pipe2(test_name2.c_str(), 2, { 0, 1 }, test_size + 1);
    size_t result;
    result = shared_pipe1.write(input_data.data(), test_size, 0);
    ASSERT_EQ(result, test_size);
    result = shared_pipe2.write(shared_pipe1.get_buffer(0), test_size, 0);
    result = shared_pipe2.read(output_data.data(), test_size, 0);
    ASSERT_EQ(result, test_size);
    ASSERT_EQ(input_data, output_data);
    // ASSERT_SUCCESS
}

TEST(CudaIPCSingle, SharedPipeOOM)
{
    const size_t test_size = 1024;
    std::vector<char> input_data(test_size);
    std::string test_name = get_thread_unique_name("SharedPipe");
    // No enough memory for input_data
    SharedPipe shared_pipe(test_name.c_str(), 1, { 0 }, test_size - 1);
    size_t result;
    result = shared_pipe.write(input_data.data(), test_size, 0);
    // Can only write test_size - 1 byte, because 1 byte is reserved for buffer tail
    ASSERT_EQ(test_size - 2, result);
}

TEST(CudaIPCSingle, SharedPipeReceiveOver)
{
    const size_t test_size = 1024;
    std::vector<uint8_t> input_data(test_size), output_data(test_size * 2);
    srand(time(nullptr));
    for (auto &c : input_data) {
        c = random() % std::numeric_limits<uint8_t>::max();
    }
    std::string test_name = get_thread_unique_name("SharedPipe");
    SharedPipe shared_pipe(test_name.c_str(), 1, { 0 }, test_size + 1);
    size_t result;
    result = shared_pipe.write(input_data.data(), test_size, 0);
    ASSERT_EQ(result, test_size);
    result = shared_pipe.read(output_data.data(), test_size * 2, 0);
    ASSERT_EQ(result, test_size);
    output_data.resize(result);
    ASSERT_EQ(input_data, output_data);
}

TEST(CudaIPCSingle, InvalidGPUMemory)
{
    std::array<char, 2048> input_data;
    memset(input_data.data(), 0, input_data.size());
    std::string test_name = get_thread_unique_name("SharedTableSimpleCopy");
    SharedTable<> table(test_name.c_str(), 1024);
    // Input data is larger than table's memory
    auto cp_result = cudaMemcpy(table.get_buffer(1), input_data.data(),
                                input_data.size(), cudaMemcpyDefault);
    ASSERT_TRUE(cp_result);
}

/**
 * @brief Try to allocate more device we have when crate SharedPipe
 * And should fail.
 */
TEST(CudaIPCSingle, InvalidDevice)
{
    std::string test_name = get_thread_unique_name("InvalidDevice");
    int gpu_count = 0;
    auto result = cudaGetDeviceCount(&gpu_count);
    ASSERT_EQ(result, cudaSuccess);

    try {
        // The device number is more than we have
        SharedPipe shared_pipe(test_name.c_str(), gpu_count + 1);
    } catch (const std::invalid_argument &e) {
        return;
    }
    FAIL();
}

/**
 * @brief Try to write to nonexistent device, which should fail
 * 
 */
TEST(CudaIPCSingle, InvalidDeviceWrite)
{
    std::string test_name = get_thread_unique_name("InvalidDeviceWrite");
    int gpu_count = 0;
    auto result = cudaGetDeviceCount(&gpu_count);
    ASSERT_EQ(result, cudaSuccess);
    SharedPipe shared_pipe(test_name.c_str(), 1, { 0 }, 1024);
    const size_t test_size = 1024;
    std::vector<uint8_t> input_data(test_size);
    try {
        // This should fail because the device id is out of range
        shared_pipe.write(input_data.data(), input_data.size(), gpu_count + 1);
    } catch (const std::invalid_argument &e) {
        return;
    }
    FAIL();
}

TEST(CudaIPCSingle, ZeroTableSize)
{
    std::string test_name = get_thread_unique_name("ZeroTableSize");
    try {
        SharedTable<> table(test_name.c_str(), 0);
    } catch (const std::invalid_argument &e) {
        return;
    }
    FAIL();
}

TEST(CudaIPCSingle, ZeroPipeSize)
{
    std::string test_name = get_thread_unique_name("ZeroPipeSize");
    try {
        SharedPipe pipe(test_name.c_str(), 1, 0);
    } catch (const std::invalid_argument &e) {
        return;
    }
    FAIL();
}