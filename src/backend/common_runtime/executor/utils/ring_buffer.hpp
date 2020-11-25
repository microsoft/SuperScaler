// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

/**
 * @brief Lock free ring buffer, single writer and single reader 
 * 
 */
class RingBuffer {
public:
    RingBuffer(size_t buffer_size);

    /**
     * @brief Push data to buffer, lock free
     * Will not copy only a part of data
     * @param data 
     * @param length 
     * @return true Success
     * @return false Failed
     */
    bool push(const void *data, size_t length);

    /**
     * @brief Pop data from ring buffer, lock free
     * Will not copy if there is no enough data in buffer
     * @param data 
     * @param length 
     * @return true Success
     * @return false Failed
     */
    bool pop(void *data, size_t length);

    /**
     * @brief How much data still in buffer
     * 
     * @return size_t 
     */
    size_t size() const;

    /**
     * @brief The total space of buffer
     * 
     * @return size_t 
     */
    size_t capacity() const;

private:
    size_t get_read_capacity() const;
    size_t get_write_capacity() const;
    volatile size_t m_head;
    volatile size_t m_tail;
    const size_t m_capacity;
    char m_data[];
};

template <class T>
class RingBufferQueue {
public:
    RingBufferQueue(size_t buffer_size_byte);
    inline bool push(const T &value);
    inline bool pop(T &value);
    inline size_t size() const;
    inline size_t capacity() const;
    inline bool empty() const;

private:
    RingBuffer m_ring_buffer;    
};

template <class T>
RingBufferQueue<T>::RingBufferQueue(size_t buffer_size_byte)
    : m_ring_buffer(buffer_size_byte)
{
}

template <class T>
inline bool RingBufferQueue<T>::push(const T &value)
{
    return m_ring_buffer.push(&value, sizeof(T));
}

template <class T>
inline bool RingBufferQueue<T>::pop(T &value)
{
    return m_ring_buffer.pop(&value, sizeof(T));
}

template <class T>
inline size_t RingBufferQueue<T>::size() const
{
    return m_ring_buffer.size() / sizeof(T);
}

template <class T>
inline size_t RingBufferQueue<T>::capacity() const
{
    return m_ring_buffer.capacity() / sizeof(T);
}

template <class T>
inline bool RingBufferQueue<T>::empty() const
{
    return (size() == 0);
}
