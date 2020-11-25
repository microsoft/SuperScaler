// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <queue>
#include <unordered_map>

#include "task.hpp"

#define ERROR_TASK_ID	0

class PollExecutor;

class TaskManager {
public:
	TaskManager(task_id_t max_task_id=UINT64_MAX);
	TaskManager(const TaskManager &) = delete;
	TaskManager operator=(const TaskManager &) = delete;
	virtual ~TaskManager();

	/**
	 * @brief Create a task
	 * @return A task id unique within the process
	 */
	template <class T, class... Args>
	task_id_t create_task(Args... args)
	{
		task_id_t t_id;
		auto task = std::make_shared<T>(args...);
		if (!task) {
			// A normal task id should be greater than 0
			return ERROR_TASK_ID;
		}
		// Task id starts from 1
		if (!m_recycle_queue.empty()) {
			t_id = m_recycle_queue.front();
			m_recycle_queue.pop();
		} else if (m_cur_max_id < m_max_task_id) {
			t_id = ++m_cur_max_id;
		} else {
			return ERROR_TASK_ID;
		}

		task->set_task_id(t_id);
		m_tasks.insert({t_id, task});
		
		return t_id;
	}

	/**
	 * @brief Delete a task
	 * @param t_id Task id
	 * @return True if successfully deleted
	 */
	bool delete_task(task_id_t t_id);
	/**
	 * @brief Get the pointer to a task
	 * @param t_id Task id
	 * @return Pointer to the task
	 */
	std::shared_ptr<Task> get_task(task_id_t t_id) const;

	/**
	 * @brief Get the number of active tasks
	 */
	size_t get_active_task_num() const;

private:
	// all created tasks stored here
	std::unordered_map<task_id_t, std::shared_ptr<Task> > m_tasks;
	std::queue<task_id_t> m_recycle_queue;
	task_id_t m_cur_max_id;
	const task_id_t m_max_task_id;
};
