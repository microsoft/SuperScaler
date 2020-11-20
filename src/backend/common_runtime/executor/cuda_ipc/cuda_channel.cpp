#include <string>
#include <sched.h>
#include <iostream>
#include <unistd.h>

#include "cuda_channel.hpp"

constexpr char CUDA_SEMAPHORE_PREFIX[] = "CudaSema_";
constexpr char CUDA_SHAREDMEMORY_PREFIX[] = "CudaMem_";
constexpr char CUDA_CHANNEL_PREFIX[] = "CudaChan_";

inline std::string cuda_get_semaphore_name(const std::string &channel_id)
{
    return std::string(CUDA_SEMAPHORE_PREFIX) + getlogin() +  channel_id;
}

inline std::string cuda_get_shared_memory_name(const std::string &channel_id)
{
    return std::string(CUDA_SHAREDMEMORY_PREFIX) + getlogin() + channel_id;
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
            if (can_access) {
                DeviceContextGuard guard(i);
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
            }
            checkCudaErrors(cudaDeviceCanAccessPeer(&can_access, j, i));
            if (can_access) {
                DeviceContextGuard guard(j);
                checkCudaErrors(cudaDeviceEnablePeerAccess(i, 0));
            }
        }
    }
}

CudaChannelSender::CudaChannelSender(const std::string &channel_id,
                                     int receiver_device_id,
                                     int sender_device_id,
                                     size_t receiver_buffer_size,
                                     size_t sender_buffer_size)
    : m_status(CudaChannelStatus::e_unconnected), m_channel_id(channel_id),
      m_receiver_device(receiver_device_id),
      m_sender_device(sender_device_id),
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

CudaChannelStatus CudaChannelSender::get_status() const
{
    return m_status;
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
        m_receiver_buffer_size, m_sender_buffer_size) <= shared_size;
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
    transfer((char *)m_handle_manager.get_address(meta.handler, m_receiver_device) +
        meta.offset, data, length);
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
    // TODO: schedule all operation of the same channel to the same worker
    // to avoid using lock here.
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
    m_received.erase(it);
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
                                         int receiver_device_id,
                                         int sender_device_id,
                                         size_t receiver_buffer_size,
                                         size_t sender_buffer_size)
    : m_status(CudaChannelStatus::e_unconnected), m_channel_id(channel_id),
      m_receiver_device(receiver_device_id),
      m_sender_device(sender_device_id)
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
    if (m_status == CudaChannelStatus::e_connected)
        return;
    std::string semaphore_name = cuda_get_semaphore_name(m_channel_id);
    //FIXME: If two receiver have the same channel_id,
    //one receiver will never know this but cannot work
    NamedSemaphore::remove(semaphore_name);
    m_semaphore.reset(
        new SemaphoreMutex(NamedSemaphore::OpenType::e_create, semaphore_name));
    m_status = CudaChannelStatus::e_connected;
}

bool CudaChannelReceiver::receive(const message_id_t &message_id,
                                  const cudaIpcMemHandle_t &handler,
                                  size_t offset, size_t length)
{
    CudaTransferMeta meta;
    meta.id = message_id;
    meta.handler = handler;
    meta.offset = offset;
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

bool CudaChannelReceiver::wait_id(const message_id_t message_id)
{
    CudaTransferAck ack;
    bool ret = false;
    do {
        ret = m_sender_fifo->pop(ack);
        if (ret) {
            m_acked.emplace(ack.id, ack);
        }
    } while (ret);
    auto it = m_acked.find(message_id);
    if (it == m_acked.end())
        return false;
    m_acked.erase(it);
    return true;
}

CudaChannelReceiver::~CudaChannelReceiver()
{
}

std::string CudaChannelReceiver::get_channel_id() const
{
    return m_channel_id;
}

CudaChannelStatus CudaChannelReceiver::get_status() const
{
    return m_status;
}

static std::string get_cuda_channel_name(int send_device, int recv_device)
{
    // Default name: PREFIX_USERNAME_sendDevice_recvDevice
    return std::string(CUDA_CHANNEL_PREFIX) + getlogin() + std::to_string(send_device) +
           std::string("_") + std::to_string(recv_device);
}

CudaSingleChannel::CudaSingleChannel(int self_device, int peer_device,
                                     size_t receiver_buffer_size, size_t sender_buffer_size)
    : m_status(CudaChannelStatus::e_unconnected),
      m_send_channel_name(get_cuda_channel_name(self_device, peer_device)),
      m_recv_channel_name(get_cuda_channel_name(peer_device, self_device))
{
    m_sender = CudaChannelSenderManager::get_manager().create_channel(
        m_send_channel_name,
        peer_device, self_device,
        receiver_buffer_size, sender_buffer_size);
    m_receiver = CudaChannelReceiverManager::get_manager().create_channel(
        m_recv_channel_name,
        self_device, peer_device,
        receiver_buffer_size, sender_buffer_size);
}

CudaSingleChannel::~CudaSingleChannel()
{
    CudaChannelSenderManager::get_manager().remove_channel(m_send_channel_name);
    CudaChannelReceiverManager::get_manager().remove_channel(m_recv_channel_name);
}

bool CudaSingleChannel::send(const message_id_t message_id, const void *buffer, size_t length)
{
    init_connection();

    bool send_result = false;
    do {
        send_result = m_sender->send(message_id, buffer, length);
    } while (!send_result);

    return true;
}

bool CudaSingleChannel::receive(const message_id_t message_id, void *buffer, size_t length)
{
    init_connection();

    cudaIpcMemHandle_t handler;
    checkCudaErrors(cudaIpcGetMemHandle(&handler, buffer));

    // Get offset of buffer's corresponding CUDA allocation
    void *base_ptr = nullptr;
    checkCuErrors(cuMemGetAddressRange(
        reinterpret_cast<CUdeviceptr *>(&base_ptr),
        nullptr, reinterpret_cast<CUdeviceptr>(buffer)));
    size_t offset = reinterpret_cast<ptrdiff_t>(buffer) -
              reinterpret_cast<ptrdiff_t>(base_ptr);

    bool receive_result = false;
    bool wait_result = false;
    do {
        receive_result = m_receiver->receive(message_id, handler, offset, length);
    } while (!receive_result);

    do {
        wait_result = m_receiver->wait_id(message_id);
    } while (!wait_result);

    return true;
}

void CudaSingleChannel::init_connection()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_status == CudaChannelStatus::e_connected)
        return;
    m_receiver->listen();

    bool connect_result = false;
    do {
        connect_result = m_sender->connect();
    } while (!connect_result);

    m_status = CudaChannelStatus::e_connected;
}

CudaChannel::CudaChannel(const rank_t self_device, const std::vector<rank_t> &devices)
{
    EnableCudaDeviceAccess();
    checkCudaErrors(cudaSetDevice(self_device));

    // Create a channel between self and every other device
    for (auto dev : devices) {
        if (dev == self_device)
            continue;
        auto it = m_channels.find(dev);
        if (it != m_channels.end())
            continue;
        m_channels.emplace(dev, std::make_shared<CudaSingleChannel>(self_device, dev));
    }
}

bool CudaChannel::send(const void *buffer, size_t length, rank_t to_rank, message_id_t message_id,
                       std::function<void(bool success, const void *buffer, size_t length)> call_back)
{
    auto it = m_channels.find(to_rank);
    bool ret;
    if (it == m_channels.end()) {
        ret = false;
    } else {
        it->second->send(message_id, buffer, length);
        ret = true;
    }
    if (call_back)
        call_back(ret, buffer, length);
    return ret;
}

bool CudaChannel::receive(void *buffer, size_t length, rank_t from_rank, message_id_t message_id,
                          std::function<void(bool success, void *buffer, size_t length)> call_back)
{
    auto it = m_channels.find(from_rank);
    bool ret;
    if (it == m_channels.end()) {
        ret = false;
    } else {
        it->second->receive(message_id, buffer, length);
        ret = true;
    }
    if (call_back)
        call_back(ret, buffer, length);
    return ret;
}
