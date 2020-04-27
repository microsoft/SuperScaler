#include "task.hpp"

Task::Task(Executor *exec, std::function<void(TaskState)> callback)
    : m_state(TaskState::e_uncommited), m_exec(exec), m_callback(callback)
{
}

Task::~Task()
{
    if (m_state == TaskState::e_unfinished) {
        wait();
    }
}

void Task::operator()()
{
    TaskState state = execute(m_exec);
    if (m_callback)
        m_callback(state);
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_state = state;
    }
    m_condition.notify_all();
}

TaskState Task::get_state() const
{
    return m_state;
}

bool Task::commit()
{
    std::lock_guard<std::mutex> lock(m_state_mutex);
    if (m_state != TaskState::e_uncommited)
        return false;
    m_state = TaskState::e_unfinished;
    return true;
}

TaskState Task::wait()
{
    std::unique_lock<std::mutex> lock(m_state_mutex);
    // Task should be commit to executor first, or it will wait forever
    if (m_state == TaskState::e_uncommited)
        return m_state;
    TaskState &state = m_state;
    m_condition.wait(lock, [&state] {
        return state == TaskState::e_success || state == TaskState::e_failed;
    });
    return m_state;
}

TaskState Task::execute(Executor *)
{
    return TaskState::e_success;
}