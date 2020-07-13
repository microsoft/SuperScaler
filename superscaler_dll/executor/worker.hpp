#pragma once

#include <thread>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>

class Task;
class WorkerScheduler;
enum class TaskState;

class Worker : public std::enable_shared_from_this<Worker> {
public:
    Worker();
    ~Worker();
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    void add_task(std::function<void(TaskState)> t);
    void add_task(std::shared_ptr<Task> t);
    void exit();
    bool is_idle() const;
    size_t get_workload() const;

private:
    void run();
    void release_self();

    std::thread m_worker_thread;
    std::queue<std::shared_ptr<Task> > m_task_queue;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::weak_ptr<WorkerScheduler> m_worker_scheduler;
    bool m_is_activated;
    bool m_is_idle;
};
