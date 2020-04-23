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
    virtual void add_task(std::shared_ptr<Task> t, bool thread_safe = true) = 0;
};