#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

#include "cuda_ipc_internal.hpp"
#include "../utils/semaphore_wrapper.hpp"
#include "../utils/shared_memory.hpp"
#include "../utils/ring_buffer.hpp"

#include "cuda_channel_defs.hpp"
#include "channel_manager.hpp"

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
    std::unique_ptr<SemaphoreMutex> m_semaphore;
    std::unique_ptr<SemaphoreLock> m_semaphore_lock;
    std::unique_ptr<SharedMemory> m_shared_memory;
    ReceiverQueue *m_receiver_fifo;
    SenderQueue *m_sender_fifo;
    std::unordered_map<message_id_t, CudaTransferMeta> m_received;
    std::mutex m_received_mutex;
    size_t m_receiver_buffer_size;
    size_t m_sender_buffer_size;
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
                 const cudaIpcMemHandle_t &handler, size_t length);
    bool wait(message_id_t &message_id);

    std::string get_channel_id() const;

private:
    CudaChannelReceiver(const std::string &channel_id,
                        size_t receive_buffer_size, size_t sender_buffer_size);

    CudaChannelStatus m_status;
    const std::string m_channel_id;
    std::unique_ptr<SemaphoreMutex> m_semaphore;
    std::unique_ptr<SharedMemory> m_shared_memory;
    ReceiverQueue *m_receiver_fifo;
    SenderQueue *m_sender_fifo;
};