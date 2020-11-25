// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <thread>
#include <gtest/gtest.h>

#include "../utils/thread_safe_queue.hpp"

TEST(ThreadSafeQueue, Basic)
{
	constexpr int n_item = 64;
	ThreadSafeQueue<int> queue(n_item);
	int i, j;

	for (i = 0; i < n_item; i++)
		queue.push(i);
	
	for (j = 0; j < n_item / 2; j++) {
		int item;
		queue.pop(item);
		ASSERT_EQ(item, j);
	}

	for (; j < n_item; j++) {
		int item = queue.pop();
		ASSERT_EQ(item, j);
	}
}

TEST(ThreadSafeQueue, PopBlock)
{
	constexpr int n_item = 64;
	ThreadSafeQueue<int> queue(n_item);
	int i, j;

	auto send_func = [&] {
		for (i = 0; i < n_item; i++)
			queue.push(i);
	};

	std::thread send_thread(send_func);
	for (j = 0; j < n_item / 2; j++) {
		int item;
		queue.pop(item);
		ASSERT_EQ(item, j);
	}

	for (; j < n_item; j++) {
		int item = queue.pop();
		ASSERT_EQ(item, j);
	}

	send_thread.join();
}

TEST(ThreadSafeQueue, PushBlock)
{
	constexpr int n_item = 64;
	ThreadSafeQueue<int> queue(n_item / 2);
	int i, j;

	auto send_func = [&] {
		for (i = 0; i < n_item; i++)
			queue.push(i);
	};

	std::thread send_thread(send_func);
	for (j = 0; j < n_item; j++) {
		int item;
		queue.pop(item);
		ASSERT_EQ(item, j);
	}

	send_thread.join();
}
