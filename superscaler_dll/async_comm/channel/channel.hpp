#pragma once

#include <memory>
#include <functional>

using rank_t = unsigned int;

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
     * @param buffer_size 
     * @param peer_rank 
     * @param callback 
     * @return true success
     * @return false failed
     */
    virtual bool send(const void *buffer, size_t buffer_size, rank_t peer_rank,
                      std::function<void(bool success, const void *buffer,
                                         size_t buffer_length)>
                          callback = nullptr) = 0;

    /**
     * @brief Receive data 
     * 
     * @param buffer 
     * @param buffer_size 
     * @param self_rank 
     * @param callback 
     * @return true success
     * @return false failed
     */
    virtual bool receive(
        void *buffer, size_t buffer_size, rank_t self_rank,
        std::function<void(bool success, void *buffer, size_t buffer_length)>
            callback = nullptr) = 0;
};