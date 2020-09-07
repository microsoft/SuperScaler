#pragma once

#include "task.hpp"
#include "cpu_kernels.hpp"
#include "gpu_kernels.hpp"

template <class T, class Func>
class ReductionTask : public Task{
public:
    ReductionTask(Executor *exec, task_callback_t callback,
            const T *buffer, T *memory, Func func,
            size_t offset, size_t num_elements);

protected:
    TaskState execute(Executor *exec) override;

private:
    const T *m_buffer;
    T *m_memory;
    Func m_func;
    size_t m_offset;
    size_t m_num_elements;
}; 


template <class T, class Func>
ReductionTask<T, Func>::ReductionTask(Executor *exec, task_callback_t callback,
                 const T *buffer, T *memory, Func func,
                 size_t offset, size_t num_elements)
    : Task(exec, callback), m_buffer(buffer), m_memory(memory), m_func(func),
      m_offset(offset), m_num_elements(num_elements)
{
}

template <class T, class Func>
TaskState ReductionTask<T, Func>::execute(Executor *) {
    m_func(m_buffer, m_memory, m_offset, m_num_elements);
    return TaskState::e_success;
}
