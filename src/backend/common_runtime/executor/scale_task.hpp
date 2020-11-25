// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "executor.hpp"
#include "task.hpp"
#include "cpu_kernels.hpp"
#include "gpu_kernels.hpp"

/**
* @brief A task for scaling or division
*/
template <class T, class ScaleImplement>
class ScaleTask : public Task{
public:
    ScaleTask(Executor *exec, task_callback_t callback,
              T *memory, T scale, ScaleImplement impl,
              size_t num_elements);

protected:
    TaskState execute(Executor *exec) override;

private:
    T *m_memory;		// data memory
    T m_scale;			// scale factor
    ScaleImplement m_impl;	// scale Implement
    size_t m_num_elements;	// number of elements in memory
};


template <class T, class ScaleImplement>
ScaleTask<T, ScaleImplement>::ScaleTask(Executor *exec, task_callback_t callback,
                              T *memory, T scale, ScaleImplement impl,
                              size_t num_elements)
    : Task(exec, callback), m_memory(memory), m_scale(scale), m_impl(impl),
      m_num_elements(num_elements)
{
}

template <class T, class ScaleImplement>
TaskState ScaleTask<T, ScaleImplement>::execute(Executor *exec) {
    // Use default stream 0 if no exec specified.
    m_impl(m_memory, m_scale, m_num_elements,
        exec == nullptr ? 0 : exec->get_context()->compute_dev_stream);
    return TaskState::e_success;
}

