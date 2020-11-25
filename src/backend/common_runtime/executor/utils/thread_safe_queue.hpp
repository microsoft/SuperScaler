// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <exception>

/**
 * @brief A thread safe queue for multiple readers and multiple writers.
 * Implemented on C++ STL queue.
 */
template <typename T>
class ThreadSafeQueue {
public:
	ThreadSafeQueue(size_t max_size = 64) : m_max_size(max_size) {}
	ThreadSafeQueue(const ThreadSafeQueue &) = delete;
	ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

	/**
	 * @brief Check if queue is full
	 * @return True if queue is full
	 */
	bool full() const
	{
		if (m_max_size == 0)
			return false;
		return m_queue.size() == m_max_size;
	}

	/**
	 * @brief Pop an element from queue, block if queue is empty
	 * @param item Contains the pop-out item
	 */
	void pop(T &item)
	{
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cond_empty.wait(lock, [this]() { return !m_queue.empty(); });
			item = m_queue.front();
			m_queue.pop();
		}
		m_cond_full.notify_one();
	}

	/**
	 * @brief Pop an element from queue, block if queue is empty
	 * @return Pop-out lement.
	 */
	T pop()
	{
		T val;
		pop(val);
		return val;
	}

	/**
	 * @brief Push an element into queue
	 * @param item Element to push in
	 */
	void push(const T &item)
	{
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cond_full.wait(lock, [this]() { return !full(); });
			m_queue.push(item);
		}
		m_cond_empty.notify_one();
	}

private:
	std::queue<T> m_queue;
	std::mutex m_mutex;
	size_t m_max_size;
	std::condition_variable m_cond_empty;
	std::condition_variable m_cond_full;
};
