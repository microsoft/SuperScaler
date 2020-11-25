// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <string>

#include <utils/shared_memory.hpp>
#include <utils.hpp>

TEST(SharedMemory, SinpleTest)
{
    std::string memory_name = get_thread_unique_name("SinpleTest");
    size_t test_size = 1024;
    SharedMemory mem(SharedMemory::OpenType::e_create, memory_name);
    mem.truncate(test_size);
    void *ptr = mem.get_ptr();
    ASSERT_NE(ptr, nullptr);
    // Test can write into the shared memory
    memset(ptr, 0, test_size);
}

TEST(SharedMemory, Remove)
{
    std::string memory_name = get_thread_unique_name("Remove");
    size_t test_size = 1024;
    {
        SharedMemory mem(SharedMemory::OpenType::e_create, memory_name);
        mem.truncate(test_size);
        // mem should be removed here
    }
    // Shared memory removed, cannot be opened
    ASSERT_THROW(
        { SharedMemory open_mem(SharedMemory::OpenType::e_open, memory_name); },
        std::runtime_error);
}

TEST(SharedMemory, Open)
{
    std::string memory_name = get_thread_unique_name("Open");
    size_t test_size = 1024;
    std::vector<char> input(test_size);
    std::vector<char> output(test_size);
    srand(time(nullptr));
    for (auto &c : input)
        c = rand();
    SharedMemory create_mem(SharedMemory::OpenType::e_create, memory_name);
    create_mem.truncate(test_size);
    void *create_ptr = create_mem.get_ptr();
    memcpy(create_ptr, input.data(), test_size);

    SharedMemory open_mem(SharedMemory::OpenType::e_open, memory_name);
    void *open_ptr = open_mem.get_ptr();
    ASSERT_NE(open_ptr, nullptr);
    memcpy(output.data(), open_ptr, test_size);
    ASSERT_EQ(input, output);
}

TEST(SharedMemory, Size)
{
    std::string memory_name = get_thread_unique_name("Size");
    size_t test_size = 1024;
    SharedMemory create_mem(SharedMemory::OpenType::e_create, memory_name);
    create_mem.truncate(test_size);

    SharedMemory open_mem(SharedMemory::OpenType::e_open, memory_name);
    size_t open_size = 0;
    bool get_result = open_mem.get_size(open_size);
    ASSERT_TRUE(get_result);
    ASSERT_EQ(open_size, test_size);
}