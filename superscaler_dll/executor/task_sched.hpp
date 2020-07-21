#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <utility>

#include "task.hpp"

class ExecInfo;
class PollExecutor;
class WorkerScheduler;

/**
 * @brief The TaskScheduler stores unfinished tasks and execution info
 * of finished tasks.
 * 
 * Life cycle of a task within the TaskScheduler:
 * 1. Executor add a task to TaskScheduler
 * 2. WorkerScheduler get a task from TaskScheduler
 * 3. Worker remove a task from TaskScheduler after it's finished
 * 4. TaskScheduler put the execution info into the finish queue
 * 5. Executor get the execution info from TaskScheduler
 */
class TaskScheduler {
public:
	TaskScheduler();
    TaskScheduler(const TaskScheduler &) = delete;
    TaskScheduler operator=(const TaskScheduler &) = delete;
	virtual ~TaskScheduler();

	/**
	 * @brief Add a task to the task pool
	 * 
	 * @param t Task to add
	 * @param thread_safe If multi-thread safe
	 * 
	 * @return True if task is successfully added
	 */
	bool add_task(std::shared_ptr<Task> t, bool thread_safe=true);

	/**
	 * @brief Add a dependence relationship that
	 * task \p who depends on task \p whom
	 */
	void add_dependence(task_id_t who, task_id_t whom);

	/**
	 * @brief Remove a finished task from the task pool, add execution info,
	 * and update dependence graph. Called by worker scheduler.
	 * 
	 * @param task_id The task id of the task to be removed
	 * 
	 * @return True if task is successfully removed
	 */
	bool remove_task(task_id_t task_id);

	/**
     * @brief Block to wait a finished task. Executor won't keep any
	 * information about this task after wait() called
     * 
     * @return The pointer to the execution info
     */
	std::shared_ptr<ExecInfo> wait();

	/**
	 * @brief Block to wait a specific task
	 * @param task_id task id
	 * @return The pointer to the execution info
	 */
	std::shared_ptr<ExecInfo> wait(task_id_t task_id);

private:
	void dispatch_runnable();

private:
	/* Task pool to store all unfinished tasks */
	std::unordered_map<task_id_t, std::pair<bool, std::shared_ptr<Task>>> m_tasks;
	std::unordered_map<task_id_t, std::vector<task_id_t> > m_dependences;
	std::unordered_map<task_id_t, size_t> m_dependence_count;
	/* Queue to store execution info of finished tasks */
	std::queue<std::shared_ptr<ExecInfo> > m_exec_infos;
	std::mutex m_task_mutex;
	std::mutex m_info_mutex;
	std::condition_variable m_condition;
	std::shared_ptr<WorkerScheduler> m_worker_scheduler;
	std::weak_ptr<PollExecutor> m_executor;
};
