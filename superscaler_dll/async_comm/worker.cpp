#include <vector>

#include "worker.hpp"
#include "task.hpp"

Worker::Worker() : m_is_activated(true), m_is_idle(true)
{
    m_worker_thread = std::thread(&Worker::run, this);
}

Worker::~Worker()
{
    exit();
}

void Worker::add_task(std::function<void(void)> t)
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
        add_task([&is_activated] { is_activated = false; });
        if (std::this_thread::get_id() != m_worker_thread.get_id()) {
            m_worker_thread.join();
        }
    }
}

bool Worker::is_idle() const
{
    return m_is_idle;
}

size_t Worker::get_workload() const
{
    return m_task_queue.size();
}

void Worker::run()
{
    while (m_is_activated || !m_task_queue.empty()) {
        std::vector<std::shared_ptr<Task> > tasks;
        // Reserve memory for tasks to optimize the push_back time
        tasks.reserve(16);
        m_is_idle = true;
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
        m_is_idle = false;
        for (auto &task : tasks) {
            if (task) {
                (*task)();
            }
        }
    }
}
