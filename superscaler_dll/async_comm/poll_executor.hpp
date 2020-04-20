#pragma once

#include <memory>
#include <list>
#include <vector>
#include <limits>
#include <mutex>

#include "executor.hpp"
#include "worker.hpp"

class Task;

class PollExecutor : public Executor {
public:
    PollExecutor(size_t m_max_worker_count = std::numeric_limits<size_t>::max());
    ~PollExecutor();

    void add_task(std::shared_ptr<Task> t, bool thread_safe = true);
private:

    void assign_task(std::shared_ptr<Task> t, bool thread_safe = true);
    void release_worker(std::weak_ptr<Worker> worker);

    size_t                                m_max_worker_count;
    std::mutex                            m_mutex;
    std::list<std::weak_ptr<Worker> >     m_idle_workers;
    std::list<std::weak_ptr<Worker> >     m_busy_workers;
    std::vector<std::shared_ptr<Worker> > m_workers;
};
