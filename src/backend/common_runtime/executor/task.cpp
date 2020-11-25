// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "task.hpp"
#include "exec_info.hpp"

Task::Task(Executor *exec, std::function<void(TaskState)> callback)
    : m_state(TaskState::e_uncommitted), m_id(0), m_exec(exec), m_callback(callback)
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

task_id_t Task::get_task_id() const
{
    return m_id;
}

void Task::set_task_id(task_id_t t_id)
{
    m_id = t_id;
}

bool Task::is_finished() const
{
    return m_state == TaskState::e_failed || m_state == TaskState::e_success;
}

bool Task::commit()
{
    std::lock_guard<std::mutex> lock(m_state_mutex);
    if (m_state != TaskState::e_uncommitted)
        return false;
    m_state = TaskState::e_unfinished;
    return true;
}

TaskState Task::wait()
{
    std::unique_lock<std::mutex> lock(m_state_mutex);
    // Task should be commit to executor first, or it will wait forever
    if (m_state == TaskState::e_uncommitted)
        return m_state;
    TaskState &state = m_state;
    m_condition.wait(lock, [&state] {
        return state == TaskState::e_success || state == TaskState::e_failed;
    });
    return m_state;
}

ExecInfo Task::gen_exec_info() const
{
    ExecState exec_state;
    if (m_state == TaskState::e_success)
        exec_state = ExecState::e_success;
    else
        exec_state = ExecState::e_fail;

    return ExecInfo(m_id, exec_state);
}

TaskState Task::execute(Executor *)
{
    return TaskState::e_success;
}