// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "../cuda_ipc/channel_manager.hpp"
#include "../cuda_ipc/cuda_channel.hpp"

constexpr char CHANNEL_NAME[] = "CudaChannelSample";
constexpr size_t receiver_buffer_size = 1024;
constexpr size_t sender_buffer_size = 512;
constexpr size_t test_size = 1024;
constexpr message_id_t message_id = 255;
constexpr int sender_device = 0;
constexpr int receiver_device = 1;
constexpr bool p2p_enable = false;

void SenderProcess()
{
    // Parper data
    std::array<char, test_size> input_data;
    for (int i = 0; i < test_size; ++i) {
        input_data[i] = (i % 128);
    }
    checkCudaErrors(cudaSetDevice(sender_device));
    void *cuda_input;
    checkCudaErrors(cudaMalloc(&cuda_input, test_size));
    checkCudaErrors(cudaMemcpy(cuda_input, input_data.data(), test_size,
                               cudaMemcpyDefault));

    // Create ChannelSender
    CudaChannelSenderManager &sender_manager =
        CudaChannelSenderManager::get_manager();
    CudaChannelSender *sender = sender_manager.create_channel(
        CHANNEL_NAME,
        receiver_device, sender_device, p2p_enable,
        receiver_buffer_size, sender_buffer_size);
    if (sender == nullptr) {
        std::cerr << "[Sender Error] Cannot open cuda channel sender\n";
        exit(-1);
    }
    std::cerr << "[Sender Info] Create cuda channel sender success\n";
    bool connect_result;

    // Connect, wait for receiver, it may return false
    do {
        connect_result = sender->connect();
    } while (!connect_result);
    std::cerr << "[Sender Info] Connect success\n";

    // Send data
    bool send_result = false;
    do {
        send_result = sender->send(message_id, cuda_input, test_size);
    } while (!send_result);
    std::cerr << "[Sender Info] Send success\n";
    // Clean up
    sender = nullptr;
    sender_manager.remove_channel(CHANNEL_NAME);
}

void ReceiverProcess()
{
    // Parper data
    std::array<char, test_size> output_data;
    void *cuda_output;
    checkCudaErrors(cudaSetDevice(receiver_device));
    checkCudaErrors(cudaMalloc(&cuda_output, test_size));
    cudaIpcMemHandle_t handler;
    checkCudaErrors(cudaIpcGetMemHandle(&handler, cuda_output));
    CudaChannelReceiverManager &receiver_manager =
        CudaChannelReceiverManager::get_manager();

    // Create ChannelReceiver
    CudaChannelReceiver *receiver = receiver_manager.create_channel(
        CHANNEL_NAME,
        receiver_device, sender_device, p2p_enable,
        receiver_buffer_size, sender_buffer_size);
    // Get connect
    receiver->listen();
    std::cerr << "[Receiver Info] Listend\n";
    // Tell sender to send data
    bool receive_result = false;
    do {
        /**
         * In this case, receive operation will not fail. However, if there is a lot of receivers
         * and the receiver's fifo is full, receive may fail.
         */
        receive_result = receiver->receive(message_id, handler, 0, test_size);
    } while (!receive_result);
    std::cerr << "[Receiver Info] receive task added\n";
    bool wait_result = false;
    message_id_t wait_id;
    do {
        wait_result = receiver->wait(wait_id);
    } while (!wait_result);
    if (wait_id != message_id) {
        std::cerr << "[Receiver Error] Wait wrong id\n";
        return;
    }
    std::cerr << "[Receiver Info] Receive success\n";
    // Clean up
    receiver = nullptr;
    receiver_manager.remove_channel(CHANNEL_NAME);
}

int main()
{
    pid_t pid = fork();
    if (pid == 0) {
        //Child
        SenderProcess();
    } else {
        ReceiverProcess();
        wait(NULL);
    }
    return 0;
}