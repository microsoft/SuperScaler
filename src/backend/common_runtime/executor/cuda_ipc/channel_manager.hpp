// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

class CudaChannelSender;
class CudaChannelReceiver;

template <class T>
class CudaChannelManager {
public:
    static CudaChannelManager &get_manager();
    T *get_channel(const std::string &channel_id);
    T *create_channel(const std::string &channel_id,
                      int receiver_device, int sender_device, bool p2p_enable,
                      size_t receiver_buffer_size, size_t sender_buffer_size);
    void remove_channel(const std::string &channel_id);

private:
    CudaChannelManager();
    static CudaChannelManager<T> *m_manager;
    std::unordered_map<std::string, T *> m_channels;
    std::mutex m_mutex;
};

template <class T>
CudaChannelManager<T> *
    CudaChannelManager<T>::m_manager = new CudaChannelManager<T>();

template <class T>
CudaChannelManager<T>::CudaChannelManager()
{
}

template <class T>
CudaChannelManager<T> &CudaChannelManager<T>::get_manager()
{
    return *m_manager;
}

template <class T>
T *CudaChannelManager<T>::get_channel(const std::string &channel_id)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto it = m_channels.find(channel_id);
    if (it == m_channels.end())
        return nullptr;
    return it->second;
}

template <class T>
T *CudaChannelManager<T>::create_channel(const std::string &channel_id,
                                         int receiver_device,
                                         int sender_device,
                                         bool p2p_enable,
                                         size_t receiver_buffer_size,
                                         size_t sender_buffer_size)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto it = m_channels.find(channel_id);
    if (it != m_channels.end()) // Already have one
        return nullptr;
    T *channel = new T(channel_id,
                       receiver_device, sender_device, p2p_enable,
                       receiver_buffer_size, sender_buffer_size);
    m_channels.insert(std::make_pair(channel_id, channel));
    return channel;
}

template <class T>
void CudaChannelManager<T>::remove_channel(const std::string &channel_id)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    auto it = m_channels.find(channel_id);
    if (it == m_channels.end())
        return;
    delete it->second;
    m_channels.erase(it);
}

using CudaChannelSenderManager = CudaChannelManager<CudaChannelSender>;
using CudaChannelReceiverManager = CudaChannelManager<CudaChannelReceiver>;