// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

#include "config.hpp"
#include "executor.hpp"
#include "task_sched.hpp"
#include "worker_sched.hpp"
#include "task_manager.hpp"
#include "worker.hpp"

enum class EventType { e_stop, e_add_task, e_add_dependence, e_task_done };

struct ExecutorEvent {
	// Event to stop PollExecutor
	ExecutorEvent()
		: m_type(EventType::e_stop) {}
	ExecutorEvent(std::shared_ptr<Task> t)
		: m_type(EventType::e_add_task), m_add_task(t) {}
	ExecutorEvent(task_id_t who, task_id_t whom)
		: m_type(EventType::e_add_dependence), m_dependence(who, whom) {}
	ExecutorEvent(task_id_t t_id)
		: m_type(EventType::e_task_done), m_finish_task(t_id) {}

	EventType m_type;
	std::shared_ptr<Task> m_add_task;
	std::pair<task_id_t, task_id_t> m_dependence;
	task_id_t m_finish_task;
};

class PollExecutor : public Executor {
public:
	friend class Worker;

	PollExecutor(compute_dev_id_t compute_dev_id = 0,
				 size_t max_worker = default_max_worker_num);
	virtual ~PollExecutor();

	/**
	 * @brief Add a task
	 * @param t Pointer to task
	 * 
	 * @return True if successfully added
	 */
	bool add_task(task_id_t t_id) override;
	/**
	 * @brief Create a task
	 * @return A task id unique within the process
	 */
	template <class T, class... Args>
	task_id_t create_task(Args... args)
	{
		return m_task_manager->create_task<T>(args...);
	}
	/**
	 * @brief Get the pointer to a task
	 * @param t_id Task id
	 * @return Pointer to the task
	 */
	std::weak_ptr<Task> get_task(task_id_t t_id) override;

	/**
	 * @brief Add a dependence relationship that
	 * task \p who depends on task \p whom
	 */
	bool add_dependence(task_id_t who, task_id_t whom) override;

	/**
	 * @brief Block to wait a finished task. Executor won't keep any
	 * information about this task after wait() called. Need to call
     * wait for every added task.
     * @return The pointer to the execution info
	 */
	ExecInfo wait() override;

	/**
	 * @brief Block to wait a specific task. Need to call wait for
     * every added task.
	 * @param task_id task id
	 * @return The pointer to the execution info
	 */
	ExecInfo wait(task_id_t task_id) override;

	/**
	 * @brief Get the pointer to the execution context.
	 * @return The pointer to the execution context.
	 */
    const ExecCtx *get_context() override;

	/**
	 * @brief Stop Poll Executor thread
	 */
	void exit();

protected:
	void notify_task_finish(task_id_t t_id) override;

private:
	void run();

private:
	std::queue<ExecutorEvent> m_event_queue;
	std::mutex m_event_mutex;
	std::condition_variable m_condition;

	std::unique_ptr<TaskManager> m_task_manager;
	std::unique_ptr<TaskScheduler> m_task_scheduler;
	std::unique_ptr<WorkerScheduler> m_worker_scheduler;

	std::thread m_executor_thread;
	bool m_is_activated;

	ExecCtx m_context;
};
