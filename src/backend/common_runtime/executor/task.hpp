// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <vector>

using task_id_t = uint64_t;

class Executor;
class ExecInfo;

enum class TaskState { e_uncommitted, e_unfinished, e_success, e_failed };

class Task : public std::enable_shared_from_this<Task> {
public:
    Task() = delete;
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;

    Task(Executor *exec, std::function<void(TaskState)> callback);
    virtual ~Task();

    void operator()();

    /**
     * @brief Change state from uncommitted to unfinished
     * 
     * @return true 
     * @return false state!=uncommited 
     */
    bool commit();
    TaskState get_state() const;
    bool is_finished() const;
    task_id_t get_task_id() const;
    void set_task_id(task_id_t t_id);

    /**
     * @brief Get dependent tasks 
     * @return vector of depend tasks
     */
    const std::vector<std::weak_ptr<Task> > &get_dependences();

    /**
     * @brief Wait until Task finished, will return directly when uncommitted
     * 
     * @return TaskState 
     */
    TaskState wait();

    /**
     * @brief Generate execution info for this task
     * @return execution info
     */
    ExecInfo gen_exec_info() const;

protected:
    /**
     * @brief Override me. 
     * 
     * @param exec 
     * @return TaskState task state after execute, success or failed
     */
    virtual TaskState execute(Executor *exec);

private:
    std::mutex m_state_mutex;
    std::condition_variable m_condition;
    TaskState m_state;
    task_id_t m_id;
    Executor *m_exec;
    // Caution: callback will be called with m_state unchanged, use the passed state instead
    std::function<void(TaskState)> m_callback;
};

using task_callback_t = std::function<void(TaskState)>;
