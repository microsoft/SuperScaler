#pragma once

#include <memory>
#include <mutex>

#include "executor.hpp"
#include "task_sched.hpp"
#include "worker_sched.hpp"
#include "task_manager.hpp"
#include "worker.hpp"

class PollExecutor : public Executor {
public:
	friend class Worker;

	PollExecutor();
	virtual ~PollExecutor();

	/**
	 * @brief Add a task
	 * @param t Pointer to task
	 * @return True if successfully added
	 */
	bool add_task(task_id_t t_id) override;
	/**
	 * @brief Create a task
	 * @return A task id unique within the process
	 */
	template <class T, class... Args>
	task_id_t create_task(Args... args);
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
	void add_dependence(task_id_t who, task_id_t whom) override;

	/**
	 * @brief Block to wait a finished task. Executor won't keep any
	 * information about this task after wait() called
     * 
     * @return The pointer to the execution info
	 */
	std::shared_ptr<ExecInfo> wait() override;

	/**
	 * @brief Block to wait a specific task
	 * @param task_id task id
	 * @return The pointer to the execution info
	 */
	std::shared_ptr<ExecInfo> wait(task_id_t task_id) override;

protected:
	virtual void notify_task_finish(task_id_t t_id) override;

private:
	std::shared_ptr<TaskScheduler> m_task_scheduler;
	std::shared_ptr<WorkerScheduler> m_worker_scheduler;
	std::unique_ptr<TaskManager> m_task_manager;
};
