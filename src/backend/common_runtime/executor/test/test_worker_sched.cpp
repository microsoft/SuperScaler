// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <gtest/gtest.h>

#include "../worker_sched.hpp"
#include "../task_manager.hpp"
#include "../poll_executor.hpp"

class MockPollExecutor : public PollExecutor {
public:
	friend class Worker;

	MockPollExecutor(std::mutex &mutex)
		: m_worker_sched(new WorkerScheduler(this)),
		  m_mutex(mutex) {}
	~MockPollExecutor() {}

public:
	std::unordered_map<task_id_t, bool> m_finished;
	std::unique_ptr<WorkerScheduler> m_worker_sched;

protected:
	void notify_task_finish(task_id_t t_id) override
	{
		// Concurrent modification, need lock to protect
		std::lock_guard<std::mutex> lock(m_mutex);
		m_finished[t_id] = true;
	}
	std::mutex &m_mutex;
};

TEST(WorkerScheduler, DispatchTask)
{
	TaskManager mgr;
	std::mutex mutex;
	MockPollExecutor exec(mutex);
	bool success = false;
	task_id_t t_id;

	t_id = mgr.create_task<Task>(
		nullptr,
		[&success](TaskState) { success = true; });
	
	auto t = mgr.get_task(t_id);
	
	ASSERT_TRUE(exec.m_worker_sched->dispatch_task(t));

	exec.m_worker_sched->stop_all_workers();

	ASSERT_TRUE(success);
	ASSERT_TRUE(exec.m_finished[t_id]);
}

TEST(WorkerScheduler, DispatchMultiplaTask)
{
	constexpr int n_tasks = 32;
	TaskManager mgr;
	std::vector<task_id_t> ids;
	std::atomic<int> count(0);
	std::mutex mutex;
	MockPollExecutor exec(mutex);

	for (int i = 0; i < n_tasks; i++) {
		auto t_id = mgr.create_task<Task>(
			nullptr,
			[&count](TaskState) { count++; });
		auto t = mgr.get_task(t_id);
		ids.push_back(t_id);

		{
			std::lock_guard<std::mutex> lock(mutex);
			exec.m_finished[t_id] = false;
		}
		ASSERT_TRUE(exec.m_worker_sched->dispatch_task(t));
	}

	exec.m_worker_sched->stop_all_workers();

	for (auto t_id : ids)
		ASSERT_TRUE(exec.m_finished[t_id]);
	ASSERT_EQ(count, n_tasks);
}
