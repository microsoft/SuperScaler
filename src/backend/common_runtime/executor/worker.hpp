// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "task.hpp"

using worker_id_t = uint64_t;

class Executor;
class WorkerScheduler;
enum class TaskState;

class Worker {
public:
    Worker(worker_id_t w_id = 0,
           Executor *executor = nullptr,
           WorkerScheduler *worker_sched = nullptr);
    ~Worker();
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    void add_task(std::function<void(TaskState)> t);
    void add_task(std::shared_ptr<Task> t);
    void exit();
    size_t get_workload() const;
    worker_id_t get_worker_id() const;

private:
    void run();
    void release_self();

    std::thread m_worker_thread;
    std::queue<std::shared_ptr<Task> > m_task_queue;
    std::mutex m_mutex;
    std::condition_variable m_condition;

    Executor *m_executor;
    WorkerScheduler *m_worker_sched;

    worker_id_t m_id;
    bool m_is_activated;
};
