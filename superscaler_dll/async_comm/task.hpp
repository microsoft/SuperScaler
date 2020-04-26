#pragma once

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

class Executor;

enum class TaskState { e_unfinished, e_success, e_failed };

class Task : public std::enable_shared_from_this<Task> {
public:
    Task() = delete;
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;

    Task(Executor *exec, std::function<void(TaskState)> callback);
    virtual ~Task();

    void operator()();
    TaskState get_state() const;
    bool is_finished() const;
    void wait();

protected:
    /**
     * @brief Override me. 
     * 
     * @param exec 
     * @return TaskState task state after execute, success or failed
     */
    virtual TaskState execute(Executor *exec);

private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    TaskState m_state;
    Executor *m_exec;
    // Caution: callback will be called with m_state unchanged, use the passwd state instead
    std::function<void(TaskState)> m_callback;
};

using task_callback_t = std::function<void(TaskState)>;
