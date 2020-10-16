#pragma once
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

#include "task.hpp"
#include "channel/channel.hpp"

class SendTask : public Task {
public:
    SendTask(Executor *exec, task_callback_t callback,
             std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
             const MemBlock &buffer);

    SendTask(Executor *exec, task_callback_t callback,
             std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
             const void *buffer, size_t buffer_length);

protected:
    TaskState execute(Executor *exec) override;

private:
    std::shared_ptr<Channel> m_channel;
    rank_t m_peer_rank;
    message_id_t m_msg_id;
    const MemBlock m_buffer;
};