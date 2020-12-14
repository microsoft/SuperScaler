// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <array>

#include "cuda_ipc/channel_manager.hpp"
#include "cuda_ipc/cuda_channel.hpp"
#include "cuda_ipc/shared_block.hpp"
#include "utils.hpp"

/**
 * Cuda Channel cannot works in single process, so here is only simple tests for channel manager
 */

TEST(CudaChannel, EnableP2PAccess)
{
    int fake_self_device = 0;
    std::vector<rank_t> fake_peer_devices = { 1 };
    int can_access = 0;

    ASSERT_EQ(
        cudaErrorPeerAccessNotEnabled,
        cudaDeviceDisablePeerAccess(static_cast<int>(fake_peer_devices[0])));

    int num_loop = 2;
    std::unique_ptr<CudaChannel> cuda_channel = nullptr;
    for (int i = 0; i < num_loop; i++) {
        cuda_channel.reset(new CudaChannel(
            static_cast<rank_t>(fake_self_device), fake_peer_devices));
        cuda_channel.reset(nullptr);
    }

    ASSERT_EQ(
        cudaSuccess,
        cudaDeviceCanAccessPeer(&can_access, fake_self_device, fake_peer_devices[0]));
    if (can_access) {
        ASSERT_EQ(
            cudaErrorPeerAccessAlreadyEnabled,
            cudaDeviceEnablePeerAccess(static_cast<int>(fake_peer_devices[0]), 0));
        ASSERT_EQ(
            cudaSuccess,
            cudaDeviceDisablePeerAccess(static_cast<int>(fake_peer_devices[0])));
    }
}

TEST(CudaChannel, ChannelReceiverManager)
{
    const std::string channel_id =
        get_thread_unique_name("ChannelReceiverManager");
    const size_t receiver_buffer_size = 1024;
    const size_t sender_buffer_size = 1024;
    int fake_send_device = 0;
    int fake_recv_device = 0;
    bool p2p_enable = false;
    CudaChannelReceiverManager &manager =
        CudaChannelReceiverManager::get_manager();
    // Create a channel receiver;
    auto create_ptr = manager.create_channel(channel_id,
                                             fake_send_device, fake_recv_device, p2p_enable,
                                             receiver_buffer_size, sender_buffer_size);
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
    int fake_send_device = 0;
    int fake_recv_device = 0;
    bool p2p_enable = false;
    CudaChannelSenderManager &manager = CudaChannelSenderManager::get_manager();
    // Create a channel receiver;
    auto create_ptr = manager.create_channel(channel_id,
                                             fake_send_device, fake_recv_device, p2p_enable,
                                             receiver_buffer_size, sender_buffer_size);
    ASSERT_NE(create_ptr, nullptr);
    // Get the channel receiver by channel id
    auto get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, create_ptr);
    // Remove the channel receiver
    manager.remove_channel(channel_id);
    get_ptr = manager.get_channel(channel_id);
    ASSERT_EQ(get_ptr, nullptr);
}
