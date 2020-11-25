// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <stdexcept>

struct FIFOItem {
    std::shared_ptr<char> m_data;
    size_t m_base;
    size_t m_size;
    FIFOItem() : m_base(0), m_size(0)
    {
    }
    FIFOItem(const FIFOItem &) = default;
};

class FIFO {
public:
    FIFO(size_t fifo_size_byte = 1024 * 1024) : m_capacity(fifo_size_byte)
    {
    }
    FIFO(const FIFO &f) : m_capacity(f.m_capacity), m_buffer(f.m_buffer)
    {
    }

    void push(const void *buffer, size_t length)
    {
        FIFOItem item;
        item.m_data = std::shared_ptr<char>(new char[length],
                                            std::default_delete<char[]>());
        item.m_size = length;
        std::memcpy(item.m_data.get(), buffer, length);
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_full_condition.wait(lock, [this] {
                return m_buffer.size() < m_capacity;
            });
            m_buffer.push(item);
        }
        m_empty_condition.notify_all();
    }
    void pop(void *buffer, size_t length)
    {
        char *to_buffer = static_cast<char *>(buffer);
        while (length > 0) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_empty_condition.wait(lock, [this] { return !m_buffer.empty(); });
            auto &item = m_buffer.front();
            if (item.m_size <= length) {
                memcpy(to_buffer, item.m_data.get() + item.m_base, item.m_size);
                to_buffer += item.m_size;
                length -= item.m_size;
                // If the length == 0, it will quite the loop next time
                m_buffer.pop();
            } else {
                memcpy(to_buffer, item.m_data.get() + item.m_base, length);
                item.m_base += length;
                item.m_size -= length;
                break;
            }
        }
        m_full_condition.notify_all();
    }

private:
    size_t m_capacity;
    std::mutex m_mutex;
    std::condition_variable m_full_condition, m_empty_condition;
    std::queue<FIFOItem> m_buffer;
};