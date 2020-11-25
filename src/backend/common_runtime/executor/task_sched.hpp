// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "config.hpp"
#include "task.hpp"
#include "exec_info.hpp"
#include "utils/thread_safe_queue.hpp"

/**
 * @brief The TaskScheduler stores unfinished tasks and execution info
 * of finished tasks.
 * 
 * Life cycle of a task within the TaskScheduler:
 * 1. Executor add a task to TaskScheduler
 * 2. Executor get a task from TaskScheduler and add to WorkerScheduler
 * 3. Executor remove a task from TaskScheduler after it's finished
 * 4. TaskScheduler put the execution info into the exec info queue
 * 5. Executor get the execution info from TaskScheduler
 */
class TaskScheduler {
public:
	friend class PollExecutor;

	TaskScheduler(size_t max_queue_size = default_exec_info_queue_size);
    TaskScheduler(const TaskScheduler &) = delete;
    TaskScheduler operator=(const TaskScheduler &) = delete;
	virtual ~TaskScheduler();

	/**
	 * @brief Add a task to the task pool
	 * @param t Task to add
	 * 
	 * @return True if task is successfully added
	 */
	bool add_task(std::shared_ptr<Task> t);

	/**
	 * @brief Add a dependence relationship that
	 * task \p who depends on task \p whom
	 */
	bool add_dependence(task_id_t who, task_id_t whom);

	/**
     * @brief Block to wait a finished task. Executor won't keep any
	 * information about this task after wait() called
     * 
     * @return The execution info
     */
	ExecInfo wait();

	/**
	 * @brief Block to wait a specific task
	 * @param task_id task id
	 * @return The execution info
	 */
	ExecInfo wait(task_id_t task_id);

	/**
	 * @brief Get all runnable tasks from runnable list
	 */
	std::shared_ptr<Task> get_runnable();

	/**
	 * @brief Remove a finished task from the task pool, add execution info,
	 * and update dependence graph.
	 * 
	 * @param task_id The task id of the task to be removed
	 * 
	 * @return True if task is successfully removed
	 */
	bool task_done(task_id_t task_id);

private:
	void sync_runnable();
	ExecInfo fetch_one_exec_info();

private:
	/* Task pool to store all unfinished tasks */
	std::unordered_map<task_id_t, std::shared_ptr<Task>> m_added_tasks;

	std::unordered_map<task_id_t, std::unordered_set<task_id_t> > m_dependences;
	std::unordered_map<task_id_t, size_t> m_dependence_count;
	std::unordered_map<task_id_t, bool> m_is_dispatched;

	// Tasks runnable
	std::list<task_id_t> m_runnable;
	// Queue to store execution info of finished tasks
	ThreadSafeQueue<ExecInfo> m_exec_info_queue;
	// Exectio info polled out from queue but not yet fetched
	std::unordered_map<task_id_t, ExecInfo> m_exec_info_wait;
};
