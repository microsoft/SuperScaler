// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_ipc_internal.hpp"
#include "../utils/semaphore_wrapper.hpp"
#include "../utils/shared_memory.hpp"
#include "../utils/ring_buffer.hpp"
#include "../channel/channel.hpp"

#include "cuda_channel_defs.hpp"
#include "channel_manager.hpp"
#include "handle_manager.hpp"

void EnableCudaDeviceAccess();

using ReceiverQueue = RingBufferQueue<CudaTransferMeta>;
using SenderQueue = RingBufferQueue<CudaTransferAck>;
using SemaphoreLock = std::lock_guard<SemaphoreMutex>;

enum class CudaChannelStatus { e_unconnected, e_connected };

class QueueAligner {
public:
    static constexpr size_t ALIGN_SIZE = 4;
    enum class Mode {
        e_create, //Create receiver and sender
        e_get // Get pointer only
    };
    QueueAligner(Mode mode, void *shared_memory, size_t receiver_buffer_size,
                 size_t sender_buffer_size);
    ReceiverQueue *get_receive_queue();
    SenderQueue *get_sender_queue();
    static size_t get_shared_memory_size(size_t receiver_buffer_size,
                                         size_t sender_buffer_size);

private:
    Mode m_mode;
    void *m_shared_memory;
    size_t m_receiver_size, m_sender_size;
};

class CudaChannelSender {
    friend class CudaChannelManager<CudaChannelSender>;

public:
    ~CudaChannelSender();
    CudaChannelSender(const CudaChannelSender &) = delete;
    CudaChannelSender &operator=(const CudaChannelSender &) = delete;

    /**
     * @brief Get connection to receiver
     *
     * @return true
     * @return false Connect failed, maybe receiver not prepared
     */
    bool connect();

    /**
     * @brief Send message to receiver synchronously
     * Can modify this function to asynchronous one to improve performance
     * @param message_id
     * @param data
     * @param length
     * @return true
     * @return false Failed, maybe because receiver not parpered
     */
    bool send(const message_id_t &message_id, const void *data, size_t length);
    CudaChannelStatus get_status() const;
    std::string get_channel_id() const;

private:
    CudaChannelSender(const std::string &channel_id,
                      int receiver_device_id, int sender_device_id,
                      size_t receiver_buffer_size, size_t sender_buffer_size);

    /**
     * @brief Transfer cuda data from \p src to \p dst synchronously
     *
     * @param dst
     * @param src
     * @param size
     */
    void transfer(void *dst, const void *src, size_t size);

    /**
     * @brief Get message meta if available
     *
     * @param message_id
     * @param meta
     * @return true
     * @return false
     */
    bool get_receiver_meta(const message_id_t &message_id,
                           CudaTransferMeta &meta);

    CudaChannelStatus m_status;
    cudaStream_t m_stream;
    const std::string m_channel_id;
    int m_receiver_device;
    int m_sender_device;
    std::unique_ptr<SemaphoreMutex> m_semaphore;
    std::unique_ptr<SemaphoreLock> m_semaphore_lock;
    std::unique_ptr<SharedMemory> m_shared_memory;
    ReceiverQueue *m_receiver_fifo;
    SenderQueue *m_sender_fifo;
    std::unordered_map<message_id_t, CudaTransferMeta> m_received;
    std::mutex m_received_mutex;
    size_t m_receiver_buffer_size;
    size_t m_sender_buffer_size;
    HandleManager m_handle_manager;
};

class CudaChannelReceiver {
    friend class CudaChannelManager<CudaChannelReceiver>;

public:
    ~CudaChannelReceiver();
    CudaChannelReceiver(const CudaChannelReceiver &) = delete;
    CudaChannelReceiver &operator=(const CudaChannelReceiver &) = delete;

    /**
     * @brief Ready to connect, can only called once
     *
     */
    void listen();

    /**
     * @brief Tell sender can send \p message_id to receiver. The sender will transfer data.
     *
     * @param message_id
     * @param handler handler for receive buffer
     * @param length
     * @return true
     * @return false
     */
    bool receive(const message_id_t &message_id,
                 const cudaIpcMemHandle_t &handler, size_t offset, size_t length);
    bool wait(message_id_t &message_id);
    bool wait_id(const message_id_t message_id);
    CudaChannelStatus get_status() const;

    std::string get_channel_id() const;

private:
    CudaChannelReceiver(const std::string &channel_id,
                        int receiver_device_id, int sender_device_id,
                        size_t receive_buffer_size, size_t sender_buffer_size);

    CudaChannelStatus m_status;
    const std::string m_channel_id;
    int m_receiver_device;
    int m_sender_device;
    std::unique_ptr<SemaphoreMutex> m_semaphore;
    std::unique_ptr<SharedMemory> m_shared_memory;
    ReceiverQueue *m_receiver_fifo;
    SenderQueue *m_sender_fifo;
    std::unordered_map<message_id_t, CudaTransferAck> m_acked;
};

/**
 * @brief A one-to-one bidirectional cuda channel
 */
class CudaSingleChannel {
public:
    CudaSingleChannel(int self_device, int peer_device,
                      size_t receiver_buffer_size=512 * sizeof(CudaTransferMeta),
                      size_t sender_buffer_size=512 * sizeof(CudaTransferAck));
    ~CudaSingleChannel();

    /**
     * @brief Send data to receiver synchronously. Will block if recv meta is not posted
     * by receiver.
     */
    bool send(const message_id_t message_id, const void *buffer, size_t length);

    /**
     * @brief Post receive meta and receive data from sender synchronously.
     * Will block if sender doesn't send data.
     */
    bool receive(const message_id_t message_id, void *buffer, size_t length);

private:
    /**
     * @brief Initialize channel connection between two sides. Connection is
     * initiated at first send or receive. This operation is atomic to prevent
     * initiate a connection multiple times.
     */
    void init_connection();

    CudaChannelStatus m_status;
    std::mutex m_mutex;

    CudaChannelSender *m_sender;
    CudaChannelReceiver *m_receiver;

    const std::string m_send_channel_name;
    const std::string m_recv_channel_name;
};

/**
 * @brief A one-to-many bidirectional cuda channel
 */
class CudaChannel : public Channel {
public:
    CudaChannel(const rank_t self_device, const std::vector<rank_t> &devices);
    ~CudaChannel() = default;

    /**
     * @brief Send data to receiver \p to_rank synchronously.
     * @param buffer Send data block, address must be gpu memory address
     * @param to_rank Rank of receiver device. Device id acts as rank.
     * @param message_id Message id if send data
     * @param call_back
     * @return True if send success. False if receiver with rank \p to_rank does not exist
     */
    bool send(const void *buffer, size_t length, rank_t to_rank, message_id_t message_id,
              std::function<void(bool success, const void *buffer, size_t length)> call_back) override;

    /**
     * @brief Receive data from sender \p from_rank synchronously. Including post recv meta and wait for
     * send completion.
     * @param buffer Recv data block, address must be gpu memory address
     * @param from_rank Rank of sender device. Device id acts as rank.
     * @param message_id Message id of receive data
     * @param call_back
     * @return True if receive success. False if sender with rank \p from_rank does not exist
     */
    bool receive(void *buffer, size_t length, rank_t from_rank, message_id_t message_id,
                 std::function<void(bool success, void *buffer, size_t length)> call_back) override;

private:
    std::map<rank_t, std::shared_ptr<CudaSingleChannel> > m_channels;
};
