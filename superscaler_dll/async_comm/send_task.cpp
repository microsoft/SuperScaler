#include "send_task.hpp"

SendTask::SendTask(Executor *exec, task_callback_t callback,
                   std::shared_ptr<Channel> channel, rank_t peer_rank,
                   const void *buffer, size_t buffer_length)
    : Task(exec, callback), m_channel(channel), m_peer_rank(peer_rank),
      m_buffer(buffer), m_buffer_length(buffer_length)
{
}

TaskState SendTask::execute(Executor *)
{
    if (!m_channel)
        return TaskState::e_failed;
    bool result =
        m_channel->send(m_buffer, m_buffer_length, m_peer_rank, nullptr);
    if (result) {
        return TaskState::e_success;
    }
    return TaskState::e_failed;
}