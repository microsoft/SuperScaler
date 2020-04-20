#include "rdma_task.hpp"
#include "rdma_tunnel.hpp"

RDMASendTask::RDMASendTask(
        Executor * exec,
        std::function<void(void)> callback,
        RDMAChannel * channel,
        unsigned int peer_rank,
        const void * tensor,
        size_t tensor_size) :
    Task(exec, callback),
    m_channel(channel), 
    m_peer_rank(peer_rank),
    m_tensor(tensor), 
    m_tensor_size (tensor_size) {
}

void RDMASendTask::execute(Executor * ) {
    auto &condition = m_condition;
    auto &mtx = m_mutex;
    bool transmission_completed = false;
    m_channel->send(m_tensor, m_tensor_size, m_peer_rank,
                      [&condition, &mtx, &transmission_completed] {
                          {
                              std::lock_guard<std::mutex> guard(mtx);
                              transmission_completed = true;
                          }
                          condition.notify_one();
                      });
    std::unique_lock<std::mutex> lock(m_mutex);
    condition.wait(lock, [&transmission_completed] { return transmission_completed; });
}

RDMAReceiveTask::RDMAReceiveTask(
        Executor * exec,
        std::function<void(void)> callback,
        RDMAChannel * channel,
        unsigned int peer_rank,
        void * tensor,
        size_t tensor_size) :
    Task(exec, callback),
    m_channel(channel), 
    m_peer_rank(peer_rank),
    m_tensor(tensor), 
    m_tensor_size (tensor_size) {
}

void RDMAReceiveTask::execute(Executor * ) {
    auto &condition = m_condition;
    auto &mtx = m_mutex;
    bool transmission_completed = false;
    m_channel->recv(m_tensor, m_tensor_size, m_peer_rank,
                      [&condition, &mtx, &transmission_completed] {
                          {
                              std::lock_guard<std::mutex> guard(mtx);
                              transmission_completed = true;
                          }
                          condition.notify_one();
                      });
    std::unique_lock<std::mutex> lock(m_mutex);
    condition.wait(lock, [&transmission_completed] { return transmission_completed; });
}