#include <gtest/gtest.h>
#include <array>

#include "cuda_ipc/channel_manager.hpp"
#include "cuda_ipc/cuda_channel.hpp"
#include "cuda_ipc/shared_block.hpp"
#include "utils.hpp"

/**
 * Cuda Channel cannot works in single process, so here is only simple tests for channel manager
 */

TEST(CudaChannel, ChannelReceiverManager)
{
    const std::string channel_id =
        get_thread_unique_name("ChannelReceiverManager");
    const size_t receiver_buffer_size = 1024;
    const size_t sender_buffer_size = 1024;
    CudaChannelReceiverManager &manager =
        CudaChannelReceiverManager::get_manager();
    // Create a channel receiver;
    auto create_ptr = manager.create_channel(channel_id, receiver_buffer_size,
                                             sender_buffer_size);
    ASSERT_NE(create_ptr, nullptr);
    // Get the channel receiver by channel id
    auto get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, create_ptr);
    // Remove the channel receiver
    manager.remove_channel(channel_id);
    get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, nullptr);
}

TEST(CudaChannel, ChannelSenderManager)
{
    const std::string channel_id =
        get_thread_unique_name("ChannelSenderManager");
    const size_t receiver_buffer_size = 1024;
    const size_t sender_buffer_size = 1024;
    CudaChannelSenderManager &manager = CudaChannelSenderManager::get_manager();
    // Create a channel receiver;
    auto create_ptr = manager.create_channel(channel_id, receiver_buffer_size,
                                             sender_buffer_size);
    ASSERT_NE(create_ptr, nullptr);
    // Get the channel receiver by channel id
    auto get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, create_ptr);
    // Remove the channel receiver
    manager.remove_channel(channel_id);
    get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, nullptr);
}