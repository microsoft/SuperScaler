// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

#include "config.hpp"
#include "task.hpp"
#include "worker.hpp"

class Executor;

class WorkerScheduler {
public:
	friend class Worker;

	WorkerScheduler(Executor *executor,
					size_t max_worker_count = default_max_worker_num);
	WorkerScheduler(const WorkerScheduler &) = delete;
	WorkerScheduler &operator=(const WorkerScheduler &) = delete;
	virtual ~WorkerScheduler();

	/**
	 * @brief Add a task to worker scheduler and let it to schedule
	 * the task to a specific worker. Called by task scheduler.
	 *
	 * @param t Pointer to task
	 * @return True if success
	 */
	bool dispatch_task(std::shared_ptr<Task> t);

	/**
	 * @brief Move a worker from busy list to idle list
	 * @param w_id Id of the worker
	 */
    void move_worker_to_idle(worker_id_t w_id);

	/**
	 * @brief Stop all workers and wait for them to exit
	 */
	void stop_all_workers();

private:
	void assign_task(std::shared_ptr<Task> t);

	Executor *m_executor;

	size_t m_max_worker_count;
	size_t m_worker_cnt;

	std::list<std::pair<worker_id_t, std::weak_ptr<Worker>>> m_idle_workers;
	std::unordered_map<worker_id_t, std::weak_ptr<Worker> > m_busy_workers;
	std::mutex m_worker_mutex;
	std::vector<std::shared_ptr<Worker>> m_workers;
};
