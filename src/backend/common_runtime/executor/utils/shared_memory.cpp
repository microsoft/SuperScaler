// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdexcept>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "shared_memory.hpp"

const int SHARED_MEMORY_MODE = 0644;

SharedMemory::SharedMemory(SharedMemory::OpenType open_type,
                           const std::string &name)
    : m_owner(false), m_length(0), m_handle(-1), m_name(name), m_ptr(nullptr)
{
    const int open_flag = O_RDWR; // Open read and write
    const int create_flag =
        O_CREAT | O_EXCL; // Create and return error if exist
    switch (open_type) {
    case OpenType::e_open:
        m_handle = shm_open(m_name.c_str(), open_flag, SHARED_MEMORY_MODE);
        break;
    case OpenType::e_create:
        m_owner = true;
        m_handle = shm_open(m_name.c_str(), open_flag | create_flag,
                            SHARED_MEMORY_MODE);
        if (m_handle >= 0) {
            fchmod(m_handle, SHARED_MEMORY_MODE);
        }
        break;
    case OpenType::e_open_or_create:
        // Try create first
        m_handle = shm_open(m_name.c_str(), open_flag | create_flag,
                            SHARED_MEMORY_MODE);
        if (m_handle >= 0) {
            m_owner = true;
            fchmod(m_handle, SHARED_MEMORY_MODE);
            break;
        }
        if (errno != EEXIST) {
            // Shared memory not exist and cannot create it, error
            break;
        }
        m_handle = shm_open(m_name.c_str(), open_flag, SHARED_MEMORY_MODE);
        break;
    default:
        // Should never get here
        throw std::invalid_argument(std::string() +
                                    "Unknow OpenType at: " + __func__);
    }
    if (m_handle < 0) {
        // Error
        throw std::runtime_error(std::string() +
                                 "Create shared memory failed, reason:" +
                                 strerror(errno) + ". At: " + __func__);
    }
    if (!m_owner) {
        // Get size for opened memory
        get_size(m_length);
    }
}

SharedMemory::~SharedMemory()
{
    if (m_handle < 0) {
        return;
    }
    if (m_ptr != nullptr && m_ptr != MAP_FAILED) {
        munmap(m_ptr, m_length);
    }
    if (m_owner)
        remove(m_name);
    close(m_handle);
}

bool SharedMemory::remove(const std::string &name) noexcept
{
    return 0 == shm_unlink(name.c_str());
}

void SharedMemory::truncate(size_t length)
{
    int ret = ftruncate(m_handle, length);
    if (ret != 0) {
        throw std::runtime_error(std::string() +
                                 "Truncate shared memory error, because: " +
                                 strerror(errno) + ". At" + __func__);
    }
    m_length = length;
}

void *SharedMemory::get_ptr()
{
    if (m_length == 0) {
        // Have not truncated, or failed
        return nullptr;
    }
    if (m_ptr != nullptr) {
        // Already opened
        return m_ptr;
    }
    const int prot_flag = PROT_READ | PROT_WRITE;
    const int shared_flag = MAP_SHARED;
    m_ptr = mmap(0, m_length, prot_flag, shared_flag, m_handle, 0);
    if (m_ptr == MAP_FAILED) {
        throw std::runtime_error(std::string() +
                                 "Cannot open shared memory, because: " +
                                 strerror(errno) + ". At: " + __func__);
    }
    return m_ptr;
}

bool SharedMemory::get_size(size_t &mem_size) const noexcept
{
    struct stat data;
    int ret = fstat(m_handle, &data);
    if (ret != 0)
        return false; // Failed
    mem_size = data.st_size;
    return true;
}