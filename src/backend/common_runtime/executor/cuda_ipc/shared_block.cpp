// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdexcept>
#include <string>

#include "shared_block.hpp"

SharedBlock::SharedBlock(size_t length) : m_length(length), m_is_internal_memory(true) {
    checkCudaErrors(cudaMalloc(&m_buffer, m_length));
    if (m_buffer == nullptr) {
        throw std::runtime_error(std::string() + "Cannot alloc memory for buffer at " + __FUNCTION_NAME__);
    }
    checkCudaErrors(cudaMemset(m_buffer, 0, m_length));
    checkCudaErrors(cudaIpcGetMemHandle(&m_handle, m_buffer));
}

SharedBlock::SharedBlock(cudaIpcMemHandle_t handle, size_t length) : m_handle(handle), m_is_internal_memory(false) {
    checkCudaErrors(cudaIpcOpenMemHandle(&m_buffer, m_handle, cudaIpcMemLazyEnablePeerAccess));
    m_length = length;
}

SharedBlock::SharedBlock(void * buffer, size_t length) : m_is_internal_memory(false) {
    if (buffer == nullptr) {
        throw std::runtime_error(std::string() + "Cannot alloc memory for buffer at " + __FUNCTION_NAME__);
    }
    checkCudaErrors(cudaIpcGetMemHandle(&m_handle, buffer));
    m_buffer = buffer;
    m_length = length;
}

SharedBlock::~SharedBlock() {
    if (is_internal_memory() && m_buffer) {
        checkCudaErrors(cudaFree(m_buffer));
        m_buffer = nullptr;
    } else {
        checkCudaErrors(cudaIpcCloseMemHandle(m_buffer));
    }
}

void * SharedBlock::get_buffer() {
    return m_buffer;
}

const void * SharedBlock::get_buffer() const {
    return m_buffer;
}

size_t SharedBlock::get_length() const {
    return m_length;
}

bool SharedBlock::is_internal_memory() const {
    return m_is_internal_memory;
}

cudaIpcMemHandle_t SharedBlock::get_handle() const {
    return m_handle;
}