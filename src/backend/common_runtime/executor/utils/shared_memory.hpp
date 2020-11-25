// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>

class SharedMemory {
public:
    enum class OpenType { e_open, e_create, e_open_or_create };

    SharedMemory(const SharedMemory &) = delete;
    SharedMemory &operator=(const SharedMemory &) = delete;

    SharedMemory(OpenType open_type, const std::string &name);
    ~SharedMemory();
    void truncate(size_t length);
    void *get_ptr();
    /**
     * @brief Get the shared memory size
     * 
     * @param mem_size 
     * @return true 
     * @return false 
     */
    bool get_size(size_t &mem_size) const noexcept;

    static bool remove(const std::string &name) noexcept;

private:
    bool m_owner;
    size_t m_length;
    int m_handle;
    std::string m_name;
    void *m_ptr;
};