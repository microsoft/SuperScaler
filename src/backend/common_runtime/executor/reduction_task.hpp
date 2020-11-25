// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "executor.hpp"
#include "task.hpp"
#include "cpu_kernels.hpp"
#include "gpu_kernels.hpp"

template <class T, class Func>
class ReductionTask : public Task{
public:
    ReductionTask(Executor *exec, task_callback_t callback,
            const T *buffer, T *memory, Func func,
            size_t num_elements);

protected:
    TaskState execute(Executor *exec) override;

private:
    const T *m_buffer;
    T *m_memory;
    Func m_func;
    size_t m_num_elements;
};


template <class T, class Func>
ReductionTask<T, Func>::ReductionTask(Executor *exec, task_callback_t callback,
                 const T *buffer, T *memory, Func func,
                 size_t num_elements)
    : Task(exec, callback), m_buffer(buffer), m_memory(memory), m_func(func),
      m_num_elements(num_elements)
{
}

template <class T, class Func>
TaskState ReductionTask<T, Func>::execute(Executor *exec) {
    // Use default stream 0 if no exec specified.
    m_func(m_buffer, m_memory, m_num_elements,
        exec == nullptr ? 0 : exec->get_context()->compute_dev_stream);
    return TaskState::e_success;
}
