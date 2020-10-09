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

    bool send(const MemBlock &buffer, rank_t to_rank, message_id_t,
              std::function<void(bool success, const MemBlock &buffer)>
                  call_back) override
    {
        auto it = m_buffers.find(to_rank);
        if (it == m_buffers.end()) {
            if (call_back)
                call_back(false, buffer);
            return false;
        }
        it->second.push(buffer.get_address(), buffer.get_length());
        if (call_back)
            call_back(true, buffer);
        return true;
    }

    bool send(const void *buffer, size_t buffer_length, rank_t to_rank, message_id_t message_id,
              std::function<void(bool success, const void *buffer, size_t buffer_length)>
                  callback = nullptr)
    {
        MemBlock mem_blk(buffer, buffer_length);

        if (callback) {
            auto bind_callback = [callback](bool success, const MemBlock &buffer) {
                callback(success, buffer.get_address(), buffer.get_length());
            };
            return send(mem_blk, to_rank, message_id, bind_callback);
        } else {
            return send(mem_blk, to_rank, message_id, nullptr);
        }
    }

    /**
     * @brief receive a tensor
     * 
     * @param tensor tensor buffer
     * @param buffer_length 
     * @param tensor_length [out] received lenght or needed length if failed
     * @param rank yourself
     * @param call_back call_back function, will be called if received successfully
     * @return if received successfully
     */
    bool receive(
        MemBlock &buffer, rank_t rank, message_id_t,
        std::function<void(bool success, MemBlock &buffer)>
            call_back) override
    {
        auto it = m_buffers.find(rank);
        if (it == m_buffers.end()) {
            if (call_back)
                call_back(false, buffer);
            return false;
        }
        it->second.pop(buffer.get_address(), buffer.get_length());
        if (call_back)
            call_back(true, buffer);
        return true;
    }

    bool receive(void *buffer, size_t buffer_length, rank_t rank, message_id_t message_id,
                 std::function<void(bool success, void *buffer, size_t buffer_length)>
                     callback = nullptr)
    {
        MemBlock mem_blk(buffer, buffer_length);
        
        if (callback) {
            auto bind_callback = [callback](bool success, MemBlock &buffer) {
                callback(success, buffer.get_address(), buffer.get_length());
            };
            return receive(mem_blk, rank, message_id, bind_callback);
        } else {
            return receive(mem_blk, rank, message_id, nullptr);
        }
    }

private:
    std::map<rank_t, FIFO> m_buffers;
};