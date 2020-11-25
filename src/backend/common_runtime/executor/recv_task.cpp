// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recv_task.hpp"

RecvTask::RecvTask(Executor *exec, task_callback_t callback,
                   std::shared_ptr<Channel> channel, rank_t peer_rank, message_id_t msg_id,
                   void *buffer, size_t length)
    : Task(exec, callback), m_channel(channel), m_peer_rank(peer_rank), m_msg_id(msg_id),
      m_buffer(buffer), m_length(length)
{
}

TaskState RecvTask::execute(Executor *)
{
    if (!m_channel)
        return TaskState::e_failed;
    bool result =
        m_channel->receive(m_buffer, m_length, m_peer_rank, m_msg_id, nullptr);
    if (result)
        return TaskState::e_success;
    return TaskState::e_failed;
}