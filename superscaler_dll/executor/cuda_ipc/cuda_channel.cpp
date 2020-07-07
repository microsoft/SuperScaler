#include <string>
#include <sched.h>
#include <iostream>

#include "cuda_channel.hpp"
#include "shared_block.hpp"

constexpr char CUDA_SEMAPHORE_PREFIX[] = "CudaSema_";
constexpr char CUDA_SHAREDMEMORY_PREFIX[] = "CudaMem_";

inline std::string cuda_get_semaphore_name(const std::string &channel_id)
{
    return std::string(CUDA_SEMAPHORE_PREFIX) + channel_id;
}

inline std::string cuda_get_shared_memory_name(const std::string &channel_id)
{
    return std::string(CUDA_SHAREDMEMORY_PREFIX) + channel_id;
}

QueueAligner::QueueAligner(Mode mode, void *ptr, size_t receiver_buffer_size,
                           size_t sender_buffer_size)
    : m_mode(mode), m_shared_memory(ptr), m_receiver_size(receiver_buffer_size),
      m_sender_size(sender_buffer_size)
{
}

ReceiverQueue *QueueAligner::get_receive_queue()
{
    if (m_mode == QueueAligner::Mode::e_create) {
        return new (m_shared_memory) ReceiverQueue(m_receiver_size);
    }
    return static_cast<ReceiverQueue *>(m_shared_memory);
}

SenderQueue *QueueAligner::get_sender_queue()
{
    char *ptr = static_cast<char *>(m_shared_memory); // Use char to get 1 byte
    size_t receiver_use = sizeof(ReceiverQueue) + m_receiver_size;
    receiver_use += ALIGN_SIZE - (receiver_use % ALIGN_SIZE);
    ptr += receiver_use;
    if (m_mode == QueueAligner::Mode::e_create) {
        return new (ptr) SenderQueue(m_sender_size);
    }
    return reinterpret_cast<SenderQueue *>(ptr);
}

size_t QueueAligner::get_shared_memory_size(size_t receiver_buffer_size,
                                            size_t sender_buffer_size)
{
    size_t receiver_use = sizeof(ReceiverQueue) + receiver_buffer_size;
    receiver_use += ALIGN_SIZE - (receiver_use % ALIGN_SIZE);
    return receiver_use + sender_buffer_size;
}

void EnableCudaDeviceAccess()
{
    int device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < device_count; ++i) {
        for (int j = i + 1; j < device_count; ++j) {
            int can_access;
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access, i, j));
            if (!can_access) {
                DeviceContextGuard guard(i);
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
            }
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access, j, i));
            if (!can_access) {
                DeviceContextGuard guard(j);
                checkCudaErrors(cudaDeviceEnablePeerAccess(i, 0));
            }
        }
    }
}

CudaChannelSender::CudaChannelSender(const std::string &channel_id,
                                     size_t receiver_buffer_size,
                                     size_t sender_buffer_size)
    : m_status(CudaChannelStatus::e_unconnected), m_channel_id(channel_id),
      m_semaphore(nullptr), m_shared_memory(nullptr), m_receiver_fifo(nullptr),
      m_sender_fifo(nullptr), m_receiver_buffer_size(receiver_buffer_size),
      m_sender_buffer_size(sender_buffer_size)
{
    checkCudaErrors(cudaStreamCreate(&m_stream));
}

CudaChannelSender::~CudaChannelSender()
{
    cudaStreamDestroy(m_stream);
}

std::string CudaChannelSender::get_channel_id() const
{
    return m_channel_id;
}

bool CudaChannelSender::connect()
{
    if (m_status == CudaChannelStatus::e_connected)
        return true;
    // Try to open semaphore
    try {
        if (m_semaphore == nullptr) {
            m_semaphore.reset(
                new SemaphoreMutex(NamedSemaphore::OpenType::e_open,
                                   cuda_get_semaphore_name(m_channel_id)));
        }
    } catch (...) {
        // If cannot open semaphore, it will throw
        m_semaphore = nullptr;
        return false;
    }
    //FIXME: If the sender would like to connect a receiver that had a peer sender,
    //       the connection operator will fail.
    std::unique_ptr<SemaphoreLock> guard(new SemaphoreLock(*m_semaphore));
    char *shared_mem = nullptr;
    if (m_shared_memory == nullptr) {
        try {
            m_shared_memory.reset(
                new SharedMemory(SharedMemory::OpenType::e_open,
                                 cuda_get_shared_memory_name(m_channel_id)));
            shared_mem = static_cast<char *>(m_shared_memory->get_ptr());
        } catch (std::runtime_error &e) {
            return false;
        }
    }
    size_t shared_size = 0;
    bool get_result = m_shared_memory->get_size(shared_size);
    if (!get_result)
        return false;
    // Check the sizeof shared memory can match the sizeof ring buffer
    bool check_result = QueueAligner::get_shared_memory_size(
        m_receiver_buffer_size, m_sender_buffer_size);
    if (!check_result) {
        return false;
    }
    QueueAligner queue_align(QueueAligner::Mode::e_get, shared_mem,
                             m_receiver_buffer_size, m_sender_buffer_size);
    m_receiver_fifo = queue_align.get_receive_queue();
    m_sender_fifo = queue_align.get_sender_queue();
    m_semaphore_lock.swap(guard);
    m_status = CudaChannelStatus::e_connected;
    return true;
}

bool CudaChannelSender::send(const message_id_t &message_id, const void *data,
                             size_t length)
{
    CudaTransferMeta meta;
    bool get_result = get_receiver_meta(message_id, meta);
    if (!get_result)
        return false;
    SharedBlock destination(meta.handler, meta.length);
    transfer(destination.get_buffer(), data, length);
    // TODO: Optimize performance by asynchronously sending acks
    bool ret = false;
    CudaTransferAck ack{ message_id };
    do {
        ret = m_sender_fifo->push(ack);
        if (!ret)
            sched_yield(); //relinquish CPU for receiver
    } while (!ret);
    return true;
}

bool CudaChannelSender::get_receiver_meta(const message_id_t &message_id,
                                          CudaTransferMeta &meta)
{
    std::lock_guard<std::mutex> lock(m_received_mutex);
    bool ret = false;
    do {
        ret = m_receiver_fifo->pop(meta);
        if (ret) {
            m_received.insert(std::make_pair(meta.id, meta));
        }
    } while (ret);
    auto it = m_received.find(message_id);
    if (it == m_received.end()) {
        return false;
    }
    meta = it->second;
    return true;
}

void CudaChannelSender::transfer(void *dst, const void *src, size_t size)
{
    if (size == 0)
        return;
    // If the transfer failed, try to set device.
    // Device setting is removed from here,
    // which I think is unnecessary
    checkCudaErrors(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, m_stream));
    checkCudaErrors(cudaStreamSynchronize(m_stream));
}

CudaChannelReceiver::CudaChannelReceiver(const std::string &channel_id,
                                         size_t receiver_buffer_size,
                                         size_t sender_buffer_size)
    : m_status(CudaChannelStatus::e_unconnected), m_channel_id(channel_id)
{
    if (channel_id.empty())
        throw std::invalid_argument(std::string() + "Empty channel id");
    SharedMemory::remove(cuda_get_shared_memory_name(channel_id));
    m_shared_memory.reset(
        new SharedMemory(SharedMemory::OpenType::e_create,
                         cuda_get_shared_memory_name(channel_id)));
    m_shared_memory->truncate(QueueAligner::get_shared_memory_size(
        receiver_buffer_size, sender_buffer_size));
    void *shared_mem = m_shared_memory->get_ptr();
    QueueAligner queue_align(QueueAligner::Mode::e_create, shared_mem,
                             receiver_buffer_size, sender_buffer_size);
    m_receiver_fifo = queue_align.get_receive_queue();
    m_sender_fifo = queue_align.get_sender_queue();
    NamedSemaphore::remove(cuda_get_semaphore_name(channel_id));
}

void CudaChannelReceiver::listen()
{
    std::string semaphore_name = cuda_get_semaphore_name(m_channel_id);
    //FIXME: If two receiver have the same channel_id,
    //one receiver will never know this but cannot work
    NamedSemaphore::remove(semaphore_name);
    m_semaphore.reset(
        new SemaphoreMutex(NamedSemaphore::OpenType::e_create, semaphore_name));
}

bool CudaChannelReceiver::receive(const message_id_t &message_id,
                                  const cudaIpcMemHandle_t &handler,
                                  size_t length)
{
    CudaTransferMeta meta;
    meta.id = message_id;
    meta.handler = handler;
    meta.length = length;
    return m_receiver_fifo->push(meta);
}

bool CudaChannelReceiver::wait(message_id_t &message_id)
{
    CudaTransferAck ack;
    bool ret = m_sender_fifo->pop(ack);
    if (ret)
        message_id = ack.id;
    return ret;
}

CudaChannelReceiver::~CudaChannelReceiver()
{
}

std::string CudaChannelReceiver::get_channel_id() const
{
    return m_channel_id;
}