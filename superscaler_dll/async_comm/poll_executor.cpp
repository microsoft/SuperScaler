#include <algorithm>

#include "poll_executor.hpp"

PollExecutor::PollExecutor(size_t max_worker_count)
    : m_max_worker_count(max_worker_count)
{
}

PollExecutor::~PollExecutor()
{
    for (auto &worker : m_workers) {
        worker->exit();
    }
}

void PollExecutor::add_task(std::shared_ptr<Task> task, bool thread_safe)
{
    if (thread_safe) {
        std::lock_guard<std::mutex> guard(m_mutex);
        assign_task(task);
    } else {
        assign_task(task);
    }
}

void PollExecutor::assign_task(std::shared_ptr<Task> t, bool thread_safe)
{
    std::weak_ptr<Worker> worker;
    if (m_idle_workers.empty()) {
        if (m_busy_workers.size() < m_max_worker_count) {
            m_workers.emplace_back(new Worker());
            worker = m_workers.back();
        } else {
            // Find the worker that has lowest workload
            auto itr =
                std::min_element(m_busy_workers.begin(), m_busy_workers.end(),
                                 [](const std::weak_ptr<Worker> &w1,
                                    const std::weak_ptr<Worker> &w2) {
                                     auto real_w1 = w1.lock();
                                     auto real_w2 = w2.lock();
                                     return real_w1->get_workload() <
                                            real_w1->get_workload();
                                 });
            worker = *itr;
            m_busy_workers.erase(itr);
        }
    } else if (thread_safe) {
        worker = m_idle_workers.back();
        m_idle_workers.pop_back();
    } else {
        std::lock_guard<std::mutex> lock(m_mutex);
        worker = m_idle_workers.back();
        m_idle_workers.pop_back();
    }
    m_busy_workers.push_back(worker);
    auto real_worker = worker.lock();
    real_worker->add_task(t);
    release_worker(worker);
}

void PollExecutor::release_worker(std::weak_ptr<Worker> worker)
{
    auto real_worker = worker.lock();
    auto &idle_workers = m_idle_workers;
    auto &busy_workers = m_busy_workers;
    auto &idle_mutex = m_mutex;
    real_worker->add_task([&idle_workers, &busy_workers, &idle_mutex, worker] {
        auto real_worker = worker.lock();
        if (!real_worker) {
            return;
        }
        std::lock_guard<std::mutex> lock(idle_mutex);
        // Remove worker from busy list
        busy_workers.remove_if([real_worker](std::weak_ptr<Worker> ptr) {
            if (auto real_ptr = ptr.lock()) {
                return real_worker == real_ptr;
            }
            return false;
        });
        // Add worker to idle list
        idle_workers.push_back(worker);
    });
}