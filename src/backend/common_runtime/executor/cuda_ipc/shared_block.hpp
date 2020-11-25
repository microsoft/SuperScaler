// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <stdio.h>

#include "cuda_ipc_internal.hpp"


class SharedBlock {
public:

    SharedBlock() = delete;
    SharedBlock(const SharedBlock &) = delete;
    SharedBlock & operator=(const SharedBlock &) = delete;

    SharedBlock(size_t length);
    SharedBlock(cudaIpcMemHandle_t handle, size_t length);
    SharedBlock(void * buffer, size_t length);
    ~SharedBlock();

    void * get_buffer();
    const void * get_buffer() const;
    size_t get_length() const;
    bool is_internal_memory() const;
    cudaIpcMemHandle_t get_handle() const;

private:
    void *              m_buffer;
    size_t              m_length;
    cudaIpcMemHandle_t  m_handle;
    bool                m_is_internal_memory;
};
