#pragma once

struct SumKernelCPUImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t offset, size_t num_elements) {
        for (size_t i = offset; i < offset+num_elements; ++i) {
            memory[i] = buffer[i] + memory[i];
        }
    }
};

struct CopyKernelCPUImpl {
    template <class T>
    void operator()(const T* buffer, T* memory, size_t offset, size_t num_elements) {
        for (size_t i = offset; i < offset+num_elements; ++i) {
            memory[i] = buffer[i];
        }
    }
};
