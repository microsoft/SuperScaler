#include "send_task.hpp"

SendTask::SendTask(Executor *exec, task_callback_t callback,
                   std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
                   const MemBlock &buffer)
    : Task(exec, callback), m_channel(channel), m_peer_rank(peer_rank), m_msg_id(msg_id),
      m_buffer(buffer)
{
}


SendTask::SendTask(Executor *exec, task_callback_t callback,
                   std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
                   const void *buffer, size_t buffer_length)
    : Task(exec, callback), m_channel(channel), m_peer_rank(peer_rank), m_msg_id(msg_id),
      m_buffer(buffer, buffer_length)
{
}

TaskState SendTask::execute(Executor *)
{
    if (!m_channel)
        return TaskState::e_failed;
    bool result =
        m_channel->send(m_buffer, m_peer_rank, m_msg_id, nullptr);
    if (result) {
        return TaskState::e_success;
    }
    return TaskState::e_failed;
}