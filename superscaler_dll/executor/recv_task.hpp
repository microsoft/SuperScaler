#pragma once
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

#include "task.hpp"
#include "channel.hpp"

class RecvTask : public Task {
public:
    RecvTask(Executor *exec, task_callback_t callback,
             std::shared_ptr<Channel> channel, rank_t self_rank, void *buffer,
             size_t buffer_length);

protected:
    TaskState execute(Executor *exec) override;

private:
    std::shared_ptr<Channel> m_channel;
    rank_t m_self_rank;
    void *m_buffer;
    size_t m_buffer_length;
};