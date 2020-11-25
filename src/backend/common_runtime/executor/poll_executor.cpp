// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <iostream>

#include "poll_executor.hpp"
#include "exec_info.hpp"

PollExecutor::PollExecutor(compute_dev_id_t compute_dev_id, size_t max_worker)
	: m_task_manager(new TaskManager),
	  m_task_scheduler(new TaskScheduler(default_exec_info_queue_size)),
	  m_worker_scheduler(new WorkerScheduler(this, max_worker)),
	  m_is_activated(true)
{
	m_context.compute_dev_id = compute_dev_id;
#ifdef HAVE_CUDA
    checkCudaErrors(cudaSetDevice(m_context.compute_dev_id));
    checkCudaErrors(cudaStreamCreate(&m_context.compute_dev_stream));
#else
    m_context.compute_dev_stream = 0;
#endif
	m_executor_thread = std::thread(&PollExecutor::run, this);
}

PollExecutor::~PollExecutor()
{
	exit();
}

bool PollExecutor::add_task(task_id_t t_id)
{
	auto task = m_task_manager->get_task(t_id);
	if (!task)
		return false;

	{
		std::lock_guard<std::mutex> lock(m_event_mutex);
		m_event_queue.emplace(task);
	}
	m_condition.notify_one();

	return true;
}

std::weak_ptr<Task> PollExecutor::get_task(task_id_t t_id)
{
	return m_task_manager->get_task(t_id);
}

bool PollExecutor::add_dependence(task_id_t who, task_id_t whom)
{
	if (!m_task_manager->get_task(who) || !m_task_manager->get_task(whom))
		return false;

	{
		std::lock_guard<std::mutex> lock(m_event_mutex);
		m_event_queue.emplace(who, whom);
	}
	m_condition.notify_one();

	return true;
}

ExecInfo PollExecutor::wait()
{
	auto exec_info = m_task_scheduler->wait();
	m_task_manager->delete_task(exec_info.get_task_id());

	return exec_info;
}

ExecInfo PollExecutor::wait(task_id_t t_id)
{
	if (!m_task_manager->get_task(t_id))
		return ExecInfo();

	auto exec_info = m_task_scheduler->wait(t_id);
	m_task_manager->delete_task(t_id);

	return exec_info;
}

const ExecCtx *PollExecutor::get_context()
{
	return &m_context;
}

void PollExecutor::exit()
{
	if (m_executor_thread.joinable()) {
		{
			std::lock_guard<std::mutex> lock(m_event_mutex);
			// Add a stop event
			m_event_queue.emplace();
		}
		m_condition.notify_one();
		if (std::this_thread::get_id() != m_executor_thread.get_id())
			m_executor_thread.join();
	}
}

void PollExecutor::notify_task_finish(task_id_t t_id)
{
	if (t_id == 0)
		return;
	{
		std::lock_guard<std::mutex> lock(m_event_mutex);
		m_event_queue.emplace(t_id);
	}
	m_condition.notify_one();
}

void PollExecutor::run()
{
	std::vector<task_id_t> tasks_to_finish;
	std::vector<std::shared_ptr<Task> > tasks_to_add;
	std::vector<std::pair<task_id_t, task_id_t> > dependence_to_add;
	
	while (m_is_activated || !m_event_queue.empty()) {
		tasks_to_finish.clear();
		tasks_to_add.clear();
		dependence_to_add.clear();

		bool if_dispatch_task = false;
		{
			std::unique_lock<std::mutex> lock(m_event_mutex);
			m_condition.wait(lock, [this] {
				return !m_event_queue.empty();
			});

			while (!m_event_queue.empty()) { 
				auto e = m_event_queue.front();
				m_event_queue.pop();

				switch (e.m_type) {
				case EventType::e_add_task:
					tasks_to_add.push_back(e.m_add_task);
					if_dispatch_task = true;
					break;
				case EventType::e_add_dependence:
					dependence_to_add.push_back(e.m_dependence);
					break;
				case EventType::e_task_done:
					tasks_to_finish.push_back(e.m_finish_task);
					if_dispatch_task = true;
					break;
				case EventType::e_stop:
					m_is_activated = false;
					break;
				default:
					break;
				}
			}
		}

		for (auto &dep : dependence_to_add)
			m_task_scheduler->add_dependence(dep.first, dep.second);

		// Tasks must be added after dependence
		for (auto &t : tasks_to_add)
			m_task_scheduler->add_task(t);
		
		for (auto &t : tasks_to_finish)
			m_task_scheduler->task_done(t);

		// Only add task and task done can result in new dispatchable tasks
		if (if_dispatch_task) {
			std::shared_ptr<Task> task_to_dispatch;
			while (task_to_dispatch = m_task_scheduler->get_runnable())
				m_worker_scheduler->dispatch_task(task_to_dispatch);
		}
	}
}
