#pragma once

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

class Executor;

class Task : public std::enable_shared_from_this<Task> {
public:
    Task() = delete;
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;

    Task(Executor *exec, std::function<void(void)> callback);
    virtual ~Task();

    void operator()();
    bool is_finished() const;
    void wait();

protected:
    virtual void execute(Executor *exec);

private:
    Executor *m_exec;
    std::function<void(void)> m_callback;

    bool m_finished;
    std::mutex m_mutex;
    std::condition_variable m_condition;
};