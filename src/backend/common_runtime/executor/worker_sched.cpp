// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <iostream>

#include "worker_sched.hpp"
#include "task_sched.hpp"
#include "worker.hpp"
#include "task.hpp"

WorkerScheduler::WorkerScheduler(Executor *executor,
								 size_t max_worker_count)
	: m_executor(executor),
	  m_max_worker_count(max_worker_count), m_worker_cnt(0)
{
	m_busy_workers.reserve(max_worker_count);
}

WorkerScheduler::~WorkerScheduler()
{
	stop_all_workers();
}

bool WorkerScheduler::dispatch_task(std::shared_ptr<Task> t)
{
	if (!t->commit())
		return false;

	{
		std::lock_guard<std::mutex> lock(m_worker_mutex);
		assign_task(t);
	}

	return true;
}

void WorkerScheduler::stop_all_workers()
{
	for (auto &worker : m_workers) {
		worker->exit();
	}
}

void WorkerScheduler::assign_task(std::shared_ptr<Task> t)
{
	// TODO: Need better strategy for scheduling send/recv tasks
	std::weak_ptr<Worker> worker;
	worker_id_t worker_id;
	if (m_idle_workers.empty()) {
		if (m_workers.size() < m_max_worker_count) {
			worker_id = ++m_worker_cnt;
			m_workers.emplace_back(new Worker(
				worker_id, m_executor, this));
			worker = m_workers.back();
			m_busy_workers.emplace(worker_id, worker);
		} else {
			// Find the worker that has lowest workload
			auto itr = std::min_element(
				m_busy_workers.begin(), m_busy_workers.end(),
				[](const std::pair<worker_id_t, std::weak_ptr<Worker>> &w1,
				   const std::pair<worker_id_t, std::weak_ptr<Worker>> &w2) {
					auto real_w1 = w1.second.lock();
					auto real_w2 = w2.second.lock();
					return real_w1->get_workload() < real_w2->get_workload();
				});
			worker = itr->second;
		}
	} else {
		worker = m_idle_workers.back().second;
		worker_id = m_idle_workers.back().first;
		m_idle_workers.pop_back();
		m_busy_workers.emplace(worker_id, worker);
	}

	auto real_worker = worker.lock();
	real_worker->add_task(t);
}

void WorkerScheduler::move_worker_to_idle(worker_id_t w_id)
{
	{
		std::lock_guard<std::mutex> lock(m_worker_mutex);
		auto worker_itr = m_busy_workers.find(w_id);
		if (worker_itr == m_busy_workers.end())
			return;
		auto worker = worker_itr->second;
		auto worker_id = worker_itr->first;

		auto real_worker = worker.lock();
		// If there is still workload in worker queue, don't move
		if (real_worker->get_workload() > 0)
			return;
		// remove worker from busy list
		m_busy_workers.erase(worker_itr);
		m_idle_workers.emplace_front(worker_id, worker);
	}
}
