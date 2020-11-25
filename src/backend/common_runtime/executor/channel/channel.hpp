// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <functional>

using rank_t = unsigned int;
using message_id_t = uint64_t;

class Channel : public std::enable_shared_from_this<Channel> {
public:
    Channel(const Channel &) = delete;
    Channel &operator=(const Channel &) = delete;

    Channel()
    {
    }

    /**
     * @brief Send data to peer rank
     *
     * @param buffer
     * @param peer_rank
     * @param callback
     * @return true success
     * @return false failed
     */
    virtual bool send(
        const void *buffer, size_t length, rank_t peer_rank, message_id_t message_id,
        std::function<void(bool success, const void *buffer, size_t length)> callback = nullptr) = 0;

    /**
     * @brief Receive data
     *
     * @param buffer
     * @param self_rank
     * @param callback
     * @return true success
     * @return false failed
     */
    virtual bool receive(
        void *buffer, size_t length, rank_t self_rank, message_id_t message_id,
        std::function<void(bool success, void *buffer, size_t length)> callback = nullptr) = 0;
};