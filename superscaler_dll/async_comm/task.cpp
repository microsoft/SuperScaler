#include "task.hpp"

Task::Task(Executor * exec, std::function<void(void)> callback) : 
    m_exec(exec), 
    m_callback(callback),
    m_finished(false) {
}

Task::~Task() {
    if (!is_finished()) {
        wait();
    }
}

void Task::operator()() {
    execute(m_exec);
    if (m_callback) {
        m_callback();
    }
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_finished = true;
    }
    m_condition.notify_all();
}

bool Task::is_finished() const {
    return m_finished;
}

void Task::wait() {
    std::unique_lock<std::mutex> m_lock(m_mutex);
    bool & finished = m_finished;
    m_condition.wait(m_lock, [&finished] { return finished; });
}

void Task::execute(Executor *) {
}
