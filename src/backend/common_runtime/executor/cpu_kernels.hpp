// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "exec_ctx.hpp"

struct SumKernelCPUImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t num_elements, compute_dev_stream_t) {
        for (size_t i = 0; i < num_elements; ++i) {
            memory[i] = buffer[i] + memory[i];
        }
    }
};

struct CopyKernelCPUImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t num_elements, compute_dev_stream_t) {
        for (size_t i = 0; i < num_elements; ++i) {
            memory[i] = buffer[i];
        }
    }
};

struct ScaleKernelCPUImpl {
    template <class T>
    void operator()(T* memory, T scale, size_t num_elements, compute_dev_stream_t) {
        for (size_t i = 0; i < num_elements; ++i) {
            memory[i] = memory[i] * scale;
        }
    }
};

struct DivKernelCPUImpl {
    template <class T>
    void operator()(T* memory, T scale, size_t num_elements, compute_dev_stream_t) {
        for (size_t i = 0; i < num_elements; ++i) {
            memory[i] = memory[i] / scale;
        }
    }
};
