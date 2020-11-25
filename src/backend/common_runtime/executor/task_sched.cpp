// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "task_sched.hpp"
#include "worker_sched.hpp"
#include "task_manager.hpp"
#include "exec_info.hpp"
#include <iostream>

TaskScheduler::TaskScheduler(size_t max_queue_size)
	: m_exec_info_queue(max_queue_size)
{
}

TaskScheduler::~TaskScheduler()
{
}

bool TaskScheduler::add_task(std::shared_ptr<Task> t)
{
	if (!m_added_tasks.insert({t->get_task_id(), t}).second)
		return false;

	// If the task is an "orphan" task, add to runnable list
	if (m_dependence_count.find(t->get_task_id()) == m_dependence_count.end()) {
		m_runnable.push_back(t->get_task_id());
	}

	return true;
}

bool TaskScheduler::add_dependence(task_id_t who, task_id_t whom)
{		
	// Can't duplicate dependence
	if (!m_dependences[whom].emplace(who).second)
		return false;

	m_dependence_count[who]++;
	// If task whom not in dependence count
	m_dependence_count.emplace(whom, 0);

	// Mark as not diapatched
	m_is_dispatched[who] = false;
	m_is_dispatched[whom] = false;

	return true;
}

ExecInfo TaskScheduler::fetch_one_exec_info()
{
	ExecInfo exec_info;
	m_exec_info_queue.pop(exec_info);
	return exec_info;
}

ExecInfo TaskScheduler::wait()
{
	ExecInfo exec_info;

	if (!m_exec_info_wait.empty()) {
		// First check if there is execution info waiting
		auto ei_it = m_exec_info_wait.begin();
		exec_info = ei_it->second;
		m_exec_info_wait.erase(ei_it);
	} else {
		// Then fetch an execution info from the queue
		exec_info = fetch_one_exec_info();
	}

	return exec_info;
}

ExecInfo TaskScheduler::wait(task_id_t task_id)
{
	ExecInfo exec_info;
	auto ei_it = m_exec_info_wait.find(task_id);
	if (ei_it != m_exec_info_wait.end()) {
		// First check if the desired task is in the wait pool
		exec_info = ei_it->second;
		m_exec_info_wait.erase(task_id);
	} else {
		/* 
		 * If not, fetch an execution info from the queue, if it's not the desired
		 * task, put it into the wait pool and fetch another from the queue
		 * until the desired task is found
		 */
		while ((exec_info = fetch_one_exec_info()).get_task_id() != task_id) { 
			m_exec_info_wait.emplace(exec_info.get_task_id(), exec_info);
		}
	}

	return exec_info;
}

void TaskScheduler::sync_runnable()
{
	int n = 0;
	for (auto t : m_dependence_count) {
		if (t.second == 0) {
			if (!m_is_dispatched[t.first]) {
				m_runnable.push_back(t.first);
			}
			n++;
		}
	}
	/*
	 * At any time, when dependence graph is not empty, there must be at least 
	 * one task that has no dependence, otherwise there is a circular dependency
	 */
	if (!m_dependence_count.empty() && n == 0)
		throw std::runtime_error("[Task Scheduler]: Circular dependency detected");
}

std::shared_ptr<Task> TaskScheduler::get_runnable()
{
	// If there is no task in runnable list, add runnable tasks to it
	if (m_runnable.empty())
		sync_runnable();

	for (auto i = m_runnable.begin(); i != m_runnable.end();) {
		std::shared_ptr<Task> task = nullptr;
		task_id_t task_id;

		auto tsk_it = m_added_tasks.find(*i);
		if (tsk_it != m_added_tasks.end()) {
			// If task has been added, dispatch it
			task = tsk_it->second;
			task_id = tsk_it->first;

			// Mark as diapatched
			m_is_dispatched[task_id] = true;
			// Remove from runnable list
			m_runnable.erase(i);
			
			return task;
		} else {
			// If task not added, check the next one
			i++;
		}
	}

	return nullptr;
}

bool TaskScheduler::task_done(task_id_t t_id)
{
	auto tsk_it = m_added_tasks.find(t_id);
	if (tsk_it == m_added_tasks.end())
		return false;

	auto dep_it = m_dependences.find(t_id);
	if (dep_it != m_dependences.end()) {
		for (auto t_succ : dep_it->second) {
			auto itr_succ = m_dependence_count.find(t_succ);
			if (itr_succ == m_dependence_count.end())
				return false;
			// Decrease dependence count on every successor of Task t_id
			if (--(itr_succ->second) == 0) {
				// Edge Trigger: only add to runnable queue when dependence
				// count decrease to 0
				m_runnable.push_back(t_succ);
			}
		}
		// Remove from dependence graph
		m_dependences.erase(dep_it);
	}

	auto depcnt_it = m_dependence_count.find(t_id);
	if (depcnt_it != m_dependence_count.end()) {
		m_dependence_count.erase(depcnt_it);
	}

	m_is_dispatched.erase(t_id);

	// Generate execution info
	auto ei = tsk_it->second->gen_exec_info();
	m_exec_info_queue.push(ei);
	
	// Remove from added tasks
	m_added_tasks.erase(t_id);

	return true;
}
