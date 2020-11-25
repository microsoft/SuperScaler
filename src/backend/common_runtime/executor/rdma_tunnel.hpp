// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <cstring>
#include <exception>
#include <utility>

class RDMAChannel {
public:
    RDMAChannel(size_t buffer_size, const std::vector<unsigned int> &ranks)
        : m_size(buffer_size)
    {
        for (auto rank : ranks) {
            if (m_buffers.find(rank) == m_buffers.end()) {
                m_buffers[rank] = std::make_pair(
                    std::shared_ptr<char>(new char[m_size],
                                          std::default_delete<char[]>()),
                    0);
            }
        }
    }

    void send(const void *tensor, size_t length, unsigned int rank,
              std::function<void(void)> callback)
    {
        if (m_buffers.find(rank) == m_buffers.end()) {
            throw std::invalid_argument("Rank is not existed.");
        }
        std::lock_guard<std::mutex> guard(m_mutex);
        if (length > (m_size - m_buffers[rank].second)) {
            throw std::invalid_argument("Length is greater than buffer size");
        }
        std::memcpy(m_buffers[rank].first.get(), tensor, length);
        m_buffers[rank].second += length;
        m_condition.notify_all();
        callback();
    }

    void recv(void *tensor, size_t length, unsigned int rank,
              std::function<void(void)> callback)
    {
        if (m_buffers.find(rank) == m_buffers.end()) {
            throw std::invalid_argument("Rank is not existed.");
        }
        if (length > m_size) {
            throw std::invalid_argument("Length is greater than buffer size");
        }
        std::unique_lock<std::mutex> lock(m_mutex);
        auto &buffers = m_buffers;
        m_condition.wait(lock, [length, rank, &buffers] {
            return length <= buffers[rank].second;
        });
        std::memcpy(tensor, m_buffers[rank].first.get(), length);
        m_buffers[rank].second -= length;
        callback();
    }

private:
    std::map<unsigned int, std::pair<std::shared_ptr<char>, size_t> > m_buffers;
    size_t m_size;
    std::mutex m_mutex;
    std::condition_variable m_condition;
};
