// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cpu_channel.hpp
 * @author Wenhao Shi(v-wenhsh@microsoft.com)
 * @brief A simple CPU channel, single process only
 * @date 2020-04-22
 */
#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <map>
#include <cstring>
#include <tuple>
#include <sstream>
#include <stdexcept>

#include "channel.hpp"
#include "fifo.hpp"

class CPUChannel : public Channel {
public:
    CPUChannel(const std::vector<rank_t> &ranks, size_t fifo_size)
    {
        for (rank_t rank : ranks) {
            auto it = m_buffers.find(rank);
            if (it != m_buffers.end())
                continue;
            m_buffers.insert({ rank, { fifo_size } });
        }
    }

    bool send(const void *buffer, size_t length, rank_t to_rank, message_id_t,
              std::function<void(bool success, const void *buffer, size_t length)>
                  call_back = nullptr) override
    {
        auto it = m_buffers.find(to_rank);
        if (it == m_buffers.end()) {
            if (call_back)
                call_back(false, buffer, length);
            return false;
        }
        it->second.push(buffer, length);
        if (call_back)
            call_back(true, buffer, length);
        return true;
    }

    /**
     * @brief receive a tensor
     *
     * @param tensor tensor buffer
     * @param length
     * @param tensor_length [out] received lenght or needed length if failed
     * @param rank yourself
     * @param call_back call_back function, will be called if received successfully
     * @return if received successfully
     */
    bool receive(
        void *buffer, size_t length, rank_t rank, message_id_t,
        std::function<void(bool success, void *buffer, size_t length)>
            call_back = nullptr) override
    {
        auto it = m_buffers.find(rank);
        if (it == m_buffers.end()) {
            if (call_back)
                call_back(false, buffer, length);
            return false;
        }
        it->second.pop(buffer, length);
        if (call_back)
            call_back(true, buffer, length);
        return true;
    }

private:
    std::map<rank_t, FIFO> m_buffers;
};