// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <thread>


#include <channel.hpp>
#include <cpu_channel.hpp>
//TODO: Modified to template test when more channels added

TEST(CPUChannel, SimpleSendReceive)
{
    CPUChannel channel({ 1 }, 100);
    bool result = false;
    result = channel.send("1", 1, 1, 0, nullptr);
    ASSERT_TRUE(result);
    result = channel.send("2", 1, 1, 0, nullptr);
    ASSERT_TRUE(result);
    result = channel.send("3", 1, 1, 0, nullptr);
    ASSERT_TRUE(result);
    char output[3];
    result = channel.receive(output, 3, 1, 0, nullptr);
    std::string output_str(output, 3), input_str("123");
    ASSERT_EQ(output_str, input_str);
}

TEST(CPUChannel, RecursiveSendRecv)
{
    CPUChannel channel({ 1 }, 100);
    bool result = false;
    // Send in call_back function
    result = channel.send("1", 1, 1, 0, [&channel](bool, const void *, size_t) {
        bool cb_result = channel.send("2", 1, 1, 0, nullptr);
        ASSERT_TRUE(cb_result);
    });
    ASSERT_TRUE(result);
    char buffer[3];
    memset(buffer, 0, 3);
    // Receive in call_back function
    result = channel.receive(
        buffer, 1, 1, 0, [&channel, &buffer](bool, void *, size_t) {
            bool cb_result = channel.receive(buffer + 1, 1, 1, 0, nullptr);
            ASSERT_TRUE(cb_result);
        });
    ASSERT_TRUE(result);
    ASSERT_STRCASEEQ(buffer, "12");
}

TEST(CPUChannel, InvalidRank)
{
    rank_t invalied_rank = 2;
    CPUChannel channel({ 1 }, 100);
    bool result = true;
    result = channel.send("1", 1, invalied_rank, 0, nullptr);
    ASSERT_FALSE(result);
    char buffer[20];
    result = channel.receive(buffer, sizeof(buffer), invalied_rank, 0, nullptr);
    ASSERT_FALSE(result);
}

TEST(CPUChannel, ReceiveOutBound)
{
    CPUChannel channel({ 1 }, 100);
    //Send 1 byte and receive 4 byte
    bool result = true;
    result = channel.send("1", 1, 1, 0, nullptr);
    ASSERT_TRUE(result);
    std::condition_variable condition;
    std::mutex mutex;
    bool success = false;
    std::thread th{ [&]() {
        char buffer[4];
        channel.receive(buffer, 4, 1, 0, nullptr);
        {
            std::lock_guard<std::mutex> lock(mutex);
            success = true;
        }
        condition.notify_all();
    } };
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto wait_result =
            condition.wait_for(lock, std::chrono::milliseconds(100));
        ASSERT_EQ(wait_result, std::cv_status::timeout);
        ASSERT_FALSE(success);
    }
    channel.send("123", 3, 1, 0, nullptr);
    th.join();
}

TEST(CPUChannelDeathTest, InvalidSendBuffer)
{
    CPUChannel channel({ 1 }, 100);
    // This will cause a segment fault
    ASSERT_DEATH({ channel.send((void *)20, 1 << 20, 1, 0, nullptr); }, "");
}

TEST(CPUChannelDeathTest, InvalidRecvBuffer)
{
    CPUChannel channel({ 1 }, 100);
    bool result = false;
    result = channel.send("1", 1, 1, 0, nullptr);
    ASSERT_TRUE(result);
    ASSERT_DEATH({ channel.receive((void *)(20), 1, 1, 0, nullptr); }, "");
}