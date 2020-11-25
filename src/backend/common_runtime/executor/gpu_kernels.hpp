// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "exec_ctx.hpp"

struct SumKernelGPUImpl
{
    template <class T>
    void operator()(const T* buffer, T* memory, size_t num_elements,
        compute_dev_stream_t stream);
};

struct SynchronizedCopyKernelImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t num_elements, compute_dev_stream_t);
};

struct ScaleKernelGPUImpl {
    template <class T>
    void operator()(T* memory, T scale, size_t num_elements,
        compute_dev_stream_t stream);
};

struct DivKernelGPUImpl {
    template <class T>
    void operator()(T* memory, T scale, size_t num_elements,
        compute_dev_stream_t stream);
};
