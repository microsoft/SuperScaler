// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <array>
#include <thread>
#include <string>
#include <chrono>
#include <utils.hpp>
#include <sys/types.h>
#include <sched.h>

#include "utils/ring_buffer.hpp"
#include "utils/shared_memory.hpp"
#include "utils.hpp"

/**
 * @brief Test ringbuffer's basic function: size, push, pop
 * 
 */
TEST(RingBuffer, Basic)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBuffer)> buffer;
    RingBuffer *ring_buffer = new (buffer.data()) RingBuffer(buffer_size);
    // Size test
    ASSERT_EQ(ring_buffer->size(), 0);
    ASSERT_EQ(ring_buffer->capacity(),
              buffer_size - 1); // 1 byte reserved for buffer tail
    srand(time(nullptr));
    int input = rand();
    bool ret = false;
    // Push test
    ret = ring_buffer->push(&input, sizeof(input));
    ASSERT_TRUE(ret);
    ASSERT_EQ(ring_buffer->size(), sizeof(input));
    // Pop test
    int output = 0;
    ret = ring_buffer->pop(&output, sizeof(output));
    ASSERT_TRUE(ret);
    ASSERT_EQ(input, output);
}

TEST(RingBuffer, SendReceive)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBuffer)> buffer;
    RingBuffer *ring_buffer = new (buffer.data()) RingBuffer(buffer_size);

    constexpr std::size_t test_size = 65536;
    for (std::size_t i = 0UL; i < test_size; ++i) {
        bool ret = false;
        do {
            ret = ring_buffer->push(&i, sizeof(i));
        } while (!ret);
        std::size_t value = 0;
        do {
            ret = ring_buffer->pop(&value, sizeof(value));
        } while (!ret);
        ASSERT_EQ(i, value);
    }
}

/**
 * @brief Send in a thread, receive in another thread
 * 
 */
TEST(RingBuffer, MeantimeSendReceive)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBuffer)> buffer;
    RingBuffer *ring_buffer = new (buffer.data()) RingBuffer(buffer_size);

    constexpr std::size_t test_size = 65536;
    auto send_func = [&] {
        for (std::size_t i = 0UL; i < test_size; ++i) {
            bool ret = false;
            do {
                ret = ring_buffer->push(&i, sizeof(i));
                if (!ret)
                    sched_yield(); //relinquish CPU for receiver
            } while (!ret);
        }
    };
    std::thread send_thread(send_func);
    for (std::size_t i = 0UL; i < test_size; ++i) {
        bool ret = false;
        std::size_t value = 0;
        do {
            ret = ring_buffer->pop(&value, sizeof(value));
            if (!ret)
                sched_yield(); // relinquish CPU for sender
        } while (!ret);
        ASSERT_EQ(i, value);
    }
    send_thread.join();
}

TEST(RingBufferQueue, Basic)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBufferQueue<std::size_t>)> buffer;
    RingBufferQueue<std::size_t> *queue =
        new (buffer.data()) RingBufferQueue<std::size_t>(buffer_size);
    std::size_t empty_size = queue->size();
    ASSERT_EQ(empty_size, 0);
    auto capacity = queue->capacity();
    ASSERT_EQ(capacity, (buffer_size - 1) / sizeof(std::size_t));
    srand(time(nullptr));
    std::size_t input = rand();
    queue->push(input);
    auto pushed_size = queue->size();
    ASSERT_EQ(pushed_size, std::size_t(1));
    std::size_t output = 0;
    queue->pop(output);
    ASSERT_EQ(input, output);
}

TEST(RingBufferQueue, SendReceive)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBufferQueue<std::size_t>)> buffer;
    RingBufferQueue<std::size_t> *queue =
        new (buffer.data()) RingBufferQueue<std::size_t>(buffer_size);

    constexpr std::size_t test_size = 65536;
    for (std::size_t i = 0UL; i < test_size; ++i) {
        bool ret = false;
        do {
            ret = queue->push(i);
            if (!ret)
                sched_yield();
        } while (!ret);
        std::size_t value = 0;
        do {
            ret = queue->pop(value);
            if (!ret)
                sched_yield();
        } while (!ret);
        ASSERT_EQ(i, value);
    }
}

TEST(RingBufferQueue, MeantimeSendReceive)
{
    constexpr std::size_t buffer_size = 64;
    std::array<char, buffer_size + sizeof(RingBufferQueue<std::size_t>)> buffer;
    RingBufferQueue<std::size_t> *queue =
        new (buffer.data()) RingBufferQueue<std::size_t>(buffer_size);

    constexpr std::size_t test_size = 65536;
    auto send_func = [&] {
        for (std::size_t i = 0UL; i < test_size; ++i) {
            bool ret = false;
            do {
                ret = queue->push(i);
                if (!ret)
                    sched_yield();
            } while (!ret);
        }
    };
    std::thread send_thread(send_func);
    for (std::size_t i = 0UL; i < test_size; ++i) {
        bool ret = false;
        std::size_t value = 0;
        do {
            ret = queue->pop(value);
            if (!ret)
                sched_yield();
        } while (!ret);
        ASSERT_EQ(i, value);
    }
    send_thread.join();
}

/**
 * @brief Use shared memory build a ring buffer, tranfer data between process
 * 
 */
TEST(RingBuffer, SharedMemory)
{
    std::string test_name = get_thread_unique_name("SharedMemoryRingBuffer");
    int pip_fd[2];
    constexpr std::size_t buffer_size = 64;
    constexpr std::size_t test_size = 65536;
    int ret = pipe(pip_fd);
    ASSERT_EQ(ret, 0);
    pid_t pid = fork();
    if (pid == 0) {
        //Child process
        int value = 0;
        int read_byte = 0;
        while (read_byte == 0) {
            read_byte +=
                read(pip_fd[0], &value, sizeof(value)); // Wait for parent ready
        }
        SharedMemory mem(SharedMemory::OpenType::e_open, test_name);
        RingBufferQueue<std::size_t> *queue =
            (RingBufferQueue<std::size_t> *)mem.get_ptr();
        for (std::size_t i = 0UL; i < test_size; ++i) {
            bool ret = false;
            do {
                ret = queue->push(i);
                if (!ret)
                    sched_yield();
            } while (!ret);
        }
        close(pip_fd[0]);
        close(pip_fd[1]);
        exit(0);
    } else {
        // Parent
        SharedMemory mem(SharedMemory::OpenType::e_create, test_name);
        mem.truncate(buffer_size + sizeof(RingBufferQueue<std::size_t>));
        RingBufferQueue<std::size_t> *queue =
            new (mem.get_ptr()) RingBufferQueue<std::size_t>(buffer_size);
        {
            // Tell child ready
            int write_byte = 0;
            int useless = 1;
            while (write_byte == 0) {
                write_byte += write(pip_fd[1], &useless, sizeof(int));
            }
        }
        for (std::size_t i = 0UL; i < test_size; ++i) {
            std::size_t value = 0;
            bool ret = false;
            do {
                ret = queue->pop(value);
                if (!ret)
                    sched_yield();
            } while (!ret);
            ASSERT_EQ(value, i);
        }
        close(pip_fd[0]);
        close(pip_fd[1]);
        int status;
        wait(&status); // Wait for child
    }
}