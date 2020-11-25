// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <atomic>
#include <unordered_set>
#include <gtest/gtest.h>

#include <iostream>

#include "../poll_executor.hpp"


TEST(PollExecutor, CreateAndGetTask)
{
	constexpr int n_tasks = 10;
	PollExecutor exec;
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = exec.create_task<Task>(nullptr, [](TaskState) {});
		ASSERT_EQ(i + 1, t_id);
		ids.push_back(t_id);
	}

	for (int i = 0; i < n_tasks; i++) {
		auto t = exec.get_task(ids[i]).lock();
		ASSERT_TRUE(t);
		ASSERT_EQ(t->get_task_id(), ids[i]);
	}
}

TEST(PollExecutor, AddAndWaitTask)
{
	PollExecutor exec;
	constexpr int n_tasks = 5;
	std::vector<task_id_t> ids;
	std::unordered_set<task_id_t> added;
	std::atomic<int> count(0);

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = exec.create_task<Task>(
			nullptr,
			[&count](TaskState) { count++; });
		ids.push_back(t_id);
		// add all task with no dependence
		ASSERT_TRUE(exec.add_task(t_id));
		added.insert(t_id);
	}

	for (int i = 0; i < n_tasks; i++) {
		auto ei = exec.wait();
		
		// Should be the task that was added before
		ASSERT_GT(added.erase(ei.get_task_id()), 0);
		ASSERT_EQ(ei.get_state(), ExecState::e_success);
		// Task should have been deleted from task manager after we fetch the exec info
		ASSERT_FALSE(exec.get_task(ei.get_task_id()).lock());
	}

	// All tasks need to be done
	ASSERT_TRUE(added.empty());
	ASSERT_EQ(count, n_tasks);
}

TEST(PollExecutor, AddDependence)
{
	PollExecutor exec;
	std::vector<task_id_t> ids;
	int count(0);
	std::unordered_map<task_id_t, int> num_pred_task;
	std::mutex mutex;

	for (int i = 0; i < 4; i++) {
		task_id_t t_id = exec.create_task<Task>(
			nullptr,
			[&num_pred_task, &mutex, &count, i](TaskState) {
				std::lock_guard<std::mutex> lock(mutex);
				// Record the num of tasks have been executed before
				num_pred_task[i] = count;
				count++;
			});
		ids.push_back(t_id);
	}

	// dependence: 0->1,2->3
	ASSERT_TRUE(exec.add_dependence(ids[1], ids[0]));
	ASSERT_TRUE(exec.add_dependence(ids[2], ids[0]));
	ASSERT_TRUE(exec.add_dependence(ids[3], ids[1]));
	ASSERT_TRUE(exec.add_dependence(ids[3], ids[2]));

	// add all tasks
	for (int i = 0; i < 4; i++)
		ASSERT_TRUE(exec.add_task(ids[i]));

	for (int i = 0; i < 4; i++)
		exec.wait();

	// No tasks should have been executed before task 0
	ASSERT_EQ(num_pred_task[0], 0);
	// At least one, at most two tasks should be executed before task 1 or 2
	ASSERT_GE(num_pred_task[1], 1);
	ASSERT_LE(num_pred_task[1], 2);
	ASSERT_GE(num_pred_task[2], 1);
	ASSERT_LE(num_pred_task[2], 2);
	// 3 tasks should have been executed before task 3
	ASSERT_EQ(num_pred_task[3], 3);
}

TEST(PollExecutor, WaitId)
{
	PollExecutor exec;
	constexpr int n_tasks = 5;
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = exec.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
		ASSERT_TRUE(exec.add_task(t_id));
	}

	for (int i = n_tasks - 1; i >= 0; i--) {
		// wait for a specific task
		auto ei = exec.wait(ids[i]);
		ASSERT_EQ(ei.get_state(), ExecState::e_success);
		ASSERT_EQ(ei.get_task_id(), ids[i]);
	} 
}
