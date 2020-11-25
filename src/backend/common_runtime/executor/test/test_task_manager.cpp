// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "../task_manager.hpp"

class FakeTask : public Task {
public:
    FakeTask(const std::string &name, std::function<void(TaskState)> callback)
        : Task(nullptr, callback), m_name(name)
    {
    }

	std::string get_name() const
	{
		return m_name;
	}

private:
	std::string m_name;
};

TEST(TaskManager, TaskCreate)
{
	constexpr int n_tests = 10;
	TaskManager mgr;
	
	for (int i = 0; i < n_tests; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			"mock_task",
			[](TaskState) {});
		// Task id should be greater than zero if successfully created
		ASSERT_EQ(i + 1, t_id);
	}
}

TEST(TaskManager, TestDelete)
{
	constexpr int n_tests = 5;
	TaskManager mgr;
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tests; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			"mock_task",
			[](TaskState) {});
		ids.push_back(t_id);
	}

	for (int i = 0; i < n_tests; i++) {
		ASSERT_TRUE(mgr.delete_task(ids[i]));
		// Test delete wrong id, expect return false
		ASSERT_FALSE(mgr.delete_task(ids[i] + n_tests));
	}
}

TEST(TaskManager, TestGet)
{
	constexpr int n_tests = 5;
	TaskManager mgr;
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tests; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ids.push_back(t_id);
	}

	for (int i = 0; i < n_tests; i++) {
		auto t = std::dynamic_pointer_cast<FakeTask>(mgr.get_task(ids[i]));
		// Test by checking task name
		ASSERT_EQ(t->get_name(), std::to_string(i) + "mock_task");
		ASSERT_EQ(t->get_task_id(), ids[i]);
		// Test get wrong task id, expect return nullptr
		ASSERT_FALSE(mgr.get_task(ids[i] + n_tests));
	}
}

TEST(TaskManager, TestOverflow)
{
	constexpr int n_tests = 10;
	TaskManager mgr(n_tests);
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tests; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ASSERT_GT(t_id, 0);
		ids.push_back(t_id);
	}

	for (int i = 0; i < n_tests; i++) {
		// exceed max task num
		task_id_t t_id = mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ASSERT_EQ(t_id, ERROR_TASK_ID);
	}

	for (int i = 0; i < 5; i++) {
		ASSERT_TRUE(mgr.delete_task(ids[i]));
	}

	for (int i = 0; i < 5; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ASSERT_EQ(t_id, ids[i]);
	}
}

TEST(TaskManager, TestTaskNum)
{
	constexpr int n_tests = 10;
	TaskManager mgr(n_tests);
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tests; i++) {
		task_id_t t_id = mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ASSERT_EQ(mgr.get_active_task_num(), i + 1);
		ids.push_back(t_id);
	}

	for (int i = 0; i < n_tests; i++) {
		// exceed max task num
		mgr.create_task<FakeTask>(
			std::to_string(i) + "mock_task",
			[](TaskState) {});
		ASSERT_EQ(mgr.get_active_task_num(), n_tests);
	}

	for (int i = 0; i < 5; i++) {
		mgr.delete_task(ids[i]);
		ASSERT_EQ(mgr.get_active_task_num(), n_tests - i - 1);
	}
}
