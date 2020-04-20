#pragma once

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <functional>

#include "task.hpp"

class RDMAChannel;

class RDMASendTask : public Task {
public:

    RDMASendTask(
        Executor * exec,
        std::function<void(void)> callback,
        RDMAChannel * channel,
        unsigned int peer_rank,
        const void * tensor,
        size_t tensor_size);

private:
    
    void execute(Executor * exec);
    
    RDMAChannel *           m_channel;
    unsigned int            m_peer_rank;
    const void *            m_tensor;
    size_t                  m_tensor_size;
    std::mutex              m_mutex;
    std::condition_variable m_condition;
};

class RDMAReceiveTask : public Task {
public:
    
    RDMAReceiveTask(
        Executor * exec,
        std::function<void(void)> callback,
        RDMAChannel * channel,
        unsigned int peer_rank,
        void * tensor,
        size_t tensor_size);

private:
    
    void execute(Executor * exec);

    RDMAChannel *           m_channel;
    unsigned int            m_peer_rank;
    void *                  m_tensor;
    size_t                  m_tensor_size;
    std::mutex              m_mutex;
    std::condition_variable m_condition;
};
