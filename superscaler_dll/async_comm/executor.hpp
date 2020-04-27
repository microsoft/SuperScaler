#pragma once

#include <memory>

class Task;

class Executor {
public:
    Executor()
    {
    }
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    virtual ~Executor(){};
    /**
     * @brief Add a task to Executor
     * 
     * @param t pointer to task
     * @param thread_safe thread_safe needed?
     * @return true add success
     * @return false failed. Maybe because task has been added before
     */
    virtual bool add_task(std::shared_ptr<Task> t, bool thread_safe = true) = 0;
};