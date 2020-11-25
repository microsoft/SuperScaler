// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <string>
#include <queue>
#include <gtest/gtest.h>
#include <unordered_set>

#include "../task_sched.hpp"
#include "../task_manager.hpp"
#include "../worker_sched.hpp"
#include "../task.cpp"

TEST(TaskScheduler, TestAddtask)
{
	TaskManager mgr;
	TaskScheduler sched;
	constexpr int n_tasks = 5;
	std::vector<task_id_t> ids;
	std::unordered_set<task_id_t> added;

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
		// add all task with no dependence
		ASSERT_TRUE(sched.add_task(mgr.get_task(t_id)));
		// can't add duplicate task
		ASSERT_FALSE(sched.add_task(mgr.get_task(t_id)));
		added.insert(t_id);
	}

	for (int i = 0; i < n_tasks; i++) {
		auto t = sched.get_runnable();
		ASSERT_TRUE(t);
		ASSERT_NE(added.find(t->get_task_id()), added.end());
	}
}

TEST(TaskScheduler, TestDependence)
{
	TaskManager mgr;
	TaskScheduler sched;
	std::vector<task_id_t> ids;
	std::shared_ptr<Task> t1, t2;

	for (int i = 0; i < 4; i++) {
		task_id_t t_id = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
	}

	// dependence: 0->1,2->3
	ASSERT_TRUE(sched.add_dependence(ids[1], ids[0]));
	// can't add duplicated dependence
	ASSERT_FALSE(sched.add_dependence(ids[1], ids[0]));
	ASSERT_TRUE(sched.add_dependence(ids[2], ids[0]));
	ASSERT_TRUE(sched.add_dependence(ids[3], ids[1]));
	ASSERT_TRUE(sched.add_dependence(ids[3], ids[2]));

	// add all tasks
	for (int i = 0; i < 4; i++) {
		ASSERT_TRUE(sched.add_task(mgr.get_task(ids[i])));
	}

	// request runnable task, expect getting task 0
	t1 = sched.get_runnable();
	ASSERT_TRUE(t1);
	ASSERT_EQ(t1->get_task_id(), ids[0]);
	// finish task 0
	ASSERT_TRUE(sched.task_done(t1->get_task_id()));

	// request runnable task, expect getting task 1,2
	t1 = sched.get_runnable();
	t2 = sched.get_runnable();
	ASSERT_TRUE(t1);
	ASSERT_TRUE(t2);
	ASSERT_TRUE((t1->get_task_id() == ids[1] && t2->get_task_id() == ids[2]) ||
				(t1->get_task_id() == ids[2] && t2->get_task_id() == ids[1]));
	// finish task 1
	sched.task_done(t1->get_task_id());

	// request runnable task, expect getting no task
	t1 = sched.get_runnable();
	ASSERT_FALSE(t1);
	// finish task 2
	sched.task_done(t2->get_task_id());
	// request runnable task, expect getting task 3
	t1 = sched.get_runnable();
	ASSERT_TRUE(t1);
	ASSERT_EQ(t1->get_task_id(), ids[3]);
	sched.task_done(t1->get_task_id());
}

TEST(TaskScheduler, TestDispatch)
{
	TaskManager mgr;
	TaskScheduler sched;
	std::shared_ptr<Task> t;

	auto id1 = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
	auto id2 = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
	
	ASSERT_TRUE(sched.add_dependence(id2, id1));

	ASSERT_TRUE(sched.add_task(mgr.get_task(id1)));
	ASSERT_TRUE(sched.add_task(mgr.get_task(id2)));

	t = sched.get_runnable();
	ASSERT_TRUE(t);
	ASSERT_EQ(t->get_task_id(), id1);
	t = sched.get_runnable();
	// task should not be dispatched twice
	ASSERT_FALSE(t);

	sched.task_done(id1);

	t = sched.get_runnable();
	ASSERT_TRUE(t);
	ASSERT_EQ(t->get_task_id(), id2);
	t = sched.get_runnable();
	// task should not be dispatched twice
	ASSERT_FALSE(t);
}

TEST(TaskScheduler, TestCircularDependency)
{
	TaskManager mgr;
	TaskScheduler sched;
	std::vector<task_id_t> ids;
	std::shared_ptr<Task> t;

	for (int i = 0; i < 4; i++) {
		task_id_t t_id = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
	}

	// dependence: 0->1<->2->3
	ASSERT_TRUE(sched.add_dependence(ids[1], ids[0]));
	ASSERT_TRUE(sched.add_dependence(ids[2], ids[1]));
	ASSERT_TRUE(sched.add_dependence(ids[1], ids[2]));
	ASSERT_TRUE(sched.add_dependence(ids[3], ids[2]));

	// add all tasks
	for (int i = 0; i < 4; i++) {
		ASSERT_TRUE(sched.add_task(mgr.get_task(ids[i])));
	}

	// request runnable task, expect getting task 0
	t = sched.get_runnable();
	ASSERT_TRUE(t);
	ASSERT_EQ(t->get_task_id(), ids[0]);
	// finish task 0
	ASSERT_TRUE(sched.task_done(t->get_task_id()));

	// expected detect circular dependency between task 1 and task 2
	ASSERT_THROW(sched.get_runnable(), std::runtime_error);
}

TEST(TaskScheduler, TestWait)
{
	TaskManager mgr;
	TaskScheduler sched;
	constexpr int n_tasks = 5;
	std::vector<task_id_t> ids;
	std::unordered_map<task_id_t, bool> finished;

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
		ASSERT_TRUE(sched.add_task(mgr.get_task(t_id)));
		finished[t_id] = false;
	}

	for (int i = 0; i < n_tasks; i++) {
		auto t = sched.get_runnable();
		// execute task
		(*t)();
		sched.task_done(t->get_task_id());
	}

	for (int i = 0; i < n_tasks; i++) {
		// wait to get execution info of all tasks
		auto ei = sched.wait();
		ASSERT_EQ(ei.get_state(), ExecState::e_success);
		finished[ei.get_task_id()] = true;
	}

	for (auto i : ids) {
		ASSERT_TRUE(finished[i]);
	}
}

TEST(TaskScheduler, TestWaitId)
{
	TaskManager mgr;
	TaskScheduler sched;
	constexpr int n_tasks = 5;
	std::vector<task_id_t> ids;

	for (int i = 0; i < n_tasks; i++) {
		task_id_t t_id = mgr.create_task<Task>(
			nullptr,
			[](TaskState) {});
		ids.push_back(t_id);
		ASSERT_TRUE(sched.add_task(mgr.get_task(t_id)));
	}

	for (int i = 0; i < n_tasks; i++) {
		auto t = sched.get_runnable();
		// execute task
		(*t)();
		sched.task_done(t->get_task_id());
	}

	for (int i = n_tasks - 1; i >= 0; i--) {
		// wait for a specific task
		auto ei = sched.wait(ids[i]);
		ASSERT_EQ(ei.get_state(), ExecState::e_success);
		ASSERT_EQ(ei.get_task_id(), ids[i]);
	}
}
