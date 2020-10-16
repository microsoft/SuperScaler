#pragma once
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

#include "task.hpp"
#include "channel/channel.hpp"

class RecvTask : public Task {
public:
    RecvTask(Executor *exec, task_callback_t callback,
             std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
             MemBlock &buffer);
    
    RecvTask(Executor *exec, task_callback_t callback,
             std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
             void *buffer, size_t buffer_length);

protected:
    TaskState execute(Executor *exec) override;

private:
    std::shared_ptr<Channel> m_channel;
    rank_t m_peer_rank;
    message_id_t m_msg_id;
    MemBlock m_buffer;
};