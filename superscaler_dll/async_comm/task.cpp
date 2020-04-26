#include "task.hpp"

Task::Task(Executor *exec, std::function<void(TaskState)> callback)
    : m_state(TaskState::e_unfinished), m_exec(exec), m_callback(callback)
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
        std::lock_guard<std::mutex> lock(m_mutex);
        m_state = state;
    }
    m_condition.notify_all();
}

TaskState Task::get_state() const
{
    return m_state;
}

void Task::wait()
{
    std::unique_lock<std::mutex> m_lock(m_mutex);
    TaskState &state = m_state;
    m_condition.wait(m_lock,
                     [&state] { return state != TaskState::e_unfinished; });
}

TaskState Task::execute(Executor *)
{
    return TaskState::e_success;
}