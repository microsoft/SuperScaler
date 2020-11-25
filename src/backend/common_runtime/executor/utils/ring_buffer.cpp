// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstring>

#include "ring_buffer.hpp"

RingBuffer::RingBuffer(size_t buffer_size)
    : m_head(0), m_tail(0), m_capacity(buffer_size)
{
}

bool RingBuffer::push(const void *data, size_t length)
{
    size_t tail = m_tail;
    const size_t write_capacity = get_write_capacity();
    if (write_capacity < length)
        return false;
    size_t next_tail = tail + length;
    // make sure pointer's add get the right value
    const char *read_ptr = reinterpret_cast<const char *>(data);
    char *write_ptr = reinterpret_cast<char *>(m_data);
    if (next_tail < m_capacity) {
        memcpy(write_ptr + tail, data, length);
    } else {
        const size_t tail_length = m_capacity - tail;

        memcpy(write_ptr + tail, read_ptr, tail_length);
        memcpy(write_ptr, read_ptr + tail_length, length - tail_length);
    }
    if (next_tail >= m_capacity)
        next_tail -= m_capacity;
    m_tail = next_tail;
    return true;
}

bool RingBuffer::pop(void *data, size_t length)
{
    size_t head = m_head;
    const size_t read_capacity = get_read_capacity();
    if (read_capacity < length)
        return false;
    const char *read_ptr = reinterpret_cast<const char *>(m_data);
    char *write_ptr = reinterpret_cast<char *>(data);
    size_t next_head = head + length;
    if (next_head < m_capacity) {
        memcpy(write_ptr, read_ptr + head, length);
    } else {
        const size_t tail_size = m_capacity - head;
        memcpy(write_ptr, read_ptr + head, tail_size);
        memcpy(write_ptr + tail_size, read_ptr, length - tail_size);
    }
    if (next_head >= m_capacity)
        next_head -= m_capacity;
    m_head = next_head;
    return true;
}

size_t RingBuffer::size() const
{
    return get_read_capacity();
}

size_t RingBuffer::capacity() const
{
    // 1 byte is reserved for ring buffer's tail
    return m_capacity - 1;
}

size_t RingBuffer::get_read_capacity() const
{
    if (m_tail >= m_head)
        return m_tail - m_head;
    return m_capacity - m_head + m_tail;
}

size_t RingBuffer::get_write_capacity() const
{
    // 1 byte reserved for tail
    if (m_tail >= m_head) {
        return m_capacity - m_tail + m_head - 1;
    }
    return m_head - m_tail - 1;
}