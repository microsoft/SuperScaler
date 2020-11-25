// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "cpu_channel.hpp"
#include "send_task.hpp"
#include "recv_task.hpp"

template <class ChannelType>
class SendRecvTest : public ::testing::Test {
public:
    SendRecvTest()
    {
        for (rank_t i = 0ull; i < 10ull; ++i)
            m_ranks.push_back(i);
    }

protected:
    void SetUp() override;
    std::shared_ptr<Channel> m_channel;
    std::vector<rank_t> m_ranks;
};

template <>
void SendRecvTest<CPUChannel>::SetUp()
{
    const size_t k_fifo_length = 1024 * 1024;
    m_channel = std::make_shared<CPUChannel>(m_ranks, k_fifo_length);
}

// ADD more channels here
using test_channels_t = ::testing::Types<CPUChannel>;
TYPED_TEST_SUITE(SendRecvTest, test_channels_t);

TYPED_TEST(SendRecvTest, SingleSendRecv)
{
    char input_str[] = "123456";
    char output_str[7];
    memset(output_str, 0, sizeof(output_str));
    SendTask send_task(nullptr, nullptr, this->m_channel, 0, 0, input_str,
                       sizeof(input_str));
    RecvTask recv_task(nullptr, nullptr, this->m_channel, 0, 0, output_str,
                       sizeof(input_str));
    send_task();
    recv_task();
    ASSERT_EQ(send_task.get_state(), TaskState::e_success);
    ASSERT_EQ(recv_task.get_state(), TaskState::e_success);
    ASSERT_STRCASEEQ(input_str, output_str);
}

TYPED_TEST(SendRecvTest, SendCallback)
{
    char input_str[] = "123456";
    char output_str[7];
    memset(output_str, 0, sizeof(output_str));

    task_callback_t send_callback = [&](TaskState state) {
        ASSERT_EQ(state, TaskState::e_success);
        RecvTask recv_task(nullptr, nullptr, this->m_channel, 0, 0, output_str,
                           sizeof(input_str));
        recv_task();
        ASSERT_EQ(recv_task.get_state(), TaskState::e_success);
    };
    SendTask send_task(nullptr, send_callback, this->m_channel, 0, 0, input_str,
                       sizeof(input_str));
    send_task();
    ASSERT_EQ(send_task.get_state(), TaskState::e_success);
    ASSERT_STRCASEEQ(input_str, output_str);
}

TYPED_TEST(SendRecvTest, RecvCallback)
{
    char input_str[] = "123456";
    char output_str[7];
    bool success = false;
    task_callback_t recv_callback = [&success](TaskState state) {
        ASSERT_EQ(state, TaskState::e_success);
        success = true;
    };
    memset(output_str, 0, sizeof(output_str));
    SendTask send_task(nullptr, nullptr, this->m_channel, 0, 0, input_str,
                       sizeof(input_str));
    RecvTask recv_task(nullptr, recv_callback, this->m_channel, 0, 0, output_str,
                       sizeof(input_str));
    send_task();
    recv_task();
    ASSERT_EQ(send_task.get_state(), TaskState::e_success);
    ASSERT_EQ(recv_task.get_state(), TaskState::e_success);
    ASSERT_TRUE(success);
    ASSERT_STRCASEEQ(input_str, output_str);
}

TYPED_TEST(SendRecvTest, RecvFirst)
{
    char input_str[] = "123456";
    char output_str[7];
    memset(output_str, 0, sizeof(output_str));
    SendTask send_task(nullptr, nullptr, this->m_channel, 0, 0, input_str,
                       sizeof(input_str));
    RecvTask recv_task(nullptr, nullptr, this->m_channel, 0, 0, output_str,
                       sizeof(input_str));
    auto recv = [&]() { recv_task(); };
    std::thread recv_thread(recv);
    send_task();
    recv_thread.join();
    ASSERT_EQ(send_task.get_state(), TaskState::e_success);
    ASSERT_EQ(recv_task.get_state(), TaskState::e_success);
    ASSERT_STRCASEEQ(input_str, output_str);
}