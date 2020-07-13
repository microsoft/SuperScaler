#pragma once

#include <memory>
#include <list>
#include <vector>
#include <mutex>

#include "task.hpp"

class Worker;
class TaskScheduler;

/**
 * @brief WorkerScheduler keeps a pool of workers, once it receives a
 * task, it will assign it to one of the workes.
 */
class WorkerScheduler {
public:
	WorkerScheduler(size_t max_worker_count);
	WorkerScheduler(const WorkerScheduler &) = delete;
	WorkerScheduler operator=(const WorkerScheduler &) = delete;
	virtual ~WorkerScheduler();

	/**
	 * @brief Add a task to worker scheduler and let it to schedule
	 * the task to a specific worker. Called by task scheduler.
	 * 
	 * @param t Pointer to task
	 * @param thread_safe If multi-thread safe
	 * @return True if success
	 */
	bool add_task(std::shared_ptr<Task> t, bool thread_safe = true);
	/**
	 * @brief Inform worker scheduler that a task is already finished,
	 * worker scheduler will then inform task scheduler about this. Called
	 * by worker
	 * 
	 * @param t_id The task id of the finished task
	 */
	void finish_task(task_id_t t_id);

private:
	friend class Worker;
    void release_worker(std::weak_ptr<Worker> worker);

private:
	void assign_task(std::shared_ptr<Task> t, bool thread_safe = true);

	size_t m_max_worker_count;
	std::mutex m_mutex;
	std::list<std::weak_ptr<Worker> > m_idle_workers;
    std::list<std::weak_ptr<Worker> > m_busy_workers;
    std::vector<std::shared_ptr<Worker> > m_workers;
	std::weak_ptr<TaskScheduler> m_task_scheduler;
};
