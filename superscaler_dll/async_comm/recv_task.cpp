#include "recv_task.hpp"

RecvTask::RecvTask(Executor *exec, task_callback_t callback,
                   std::shared_ptr<Channel> channel, rank_t self_rank,
                   void *buffer, size_t buffer_length)
    : Task(exec, callback), m_channel(channel), m_self_rank(self_rank),
      m_buffer(buffer), m_buffer_length(buffer_length)
{
}

TaskState RecvTask::execute(Executor *)
{
    if (!m_channel)
        return TaskState::e_failed;
    bool result =
        m_channel->receive(m_buffer, m_buffer_length, m_self_rank, nullptr);
    if (result)
        return TaskState::e_success;
    return TaskState::e_failed;
}