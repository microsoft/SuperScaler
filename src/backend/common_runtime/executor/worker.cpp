// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>

#include "worker.hpp"
#include "task.hpp"
#include "worker_sched.hpp"
#include "executor.hpp"

Worker::Worker(worker_id_t id, Executor *executor, WorkerScheduler *worker_sched)
    : m_executor(executor), m_worker_sched(worker_sched),
      m_id(id), m_is_activated(true)
{
    m_worker_thread = std::thread(&Worker::run, this);
}

Worker::~Worker()
{
    exit();
}

void Worker::add_task(std::function<void(TaskState)> t)
{
    add_task(std::make_shared<Task>(nullptr, t));
}

void Worker::add_task(std::shared_ptr<Task> t)
{
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_task_queue.push(t);
    }
    m_condition.notify_one();
}

void Worker::exit()
{
    if (m_worker_thread.joinable()) {
        bool &is_activated = m_is_activated;
        add_task([&is_activated](TaskState) { is_activated = false; });
        if (std::this_thread::get_id() != m_worker_thread.get_id()) {
            m_worker_thread.join();
        }
    }
}

size_t Worker::get_workload() const
{
    return m_task_queue.size();
}

worker_id_t Worker::get_worker_id() const
{
    return m_id;
}

void Worker::run()
{
#ifdef HAVE_CUDA
    if (m_executor)
        checkCudaErrors(cudaSetDevice(m_executor->get_context()->compute_dev_id));
#endif

    while (m_is_activated || !m_task_queue.empty()) {
        std::vector<std::shared_ptr<Task> > tasks;
        // Reserve memory for tasks to optimize the push_back time
        tasks.reserve(16);
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            auto &task_queue = m_task_queue;
            m_condition.wait(lock,
                             [&task_queue] { return !task_queue.empty(); });
            while (!m_task_queue.empty()) {
                tasks.push_back(m_task_queue.front());
                m_task_queue.pop();
            }
        }
        for (auto &task : tasks) {
            if (task) {
                (*task)();
                if (m_executor)
                    m_executor->notify_task_finish(task->get_task_id());
            }
        }

        if (m_task_queue.empty()) {
            if (m_worker_sched)
                m_worker_sched->move_worker_to_idle(m_id);
        }
    }
}
