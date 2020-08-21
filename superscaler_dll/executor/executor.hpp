#pragma once

#include <memory>

#include "task.hpp"
#include "exec_info.hpp"

class Executor {
public:
    friend class Worker;

    Executor()
    {
    }
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    virtual ~Executor(){};
    
    /**
     * @brief Add a task to Executor
     * 
     * @param t pointer to task
     * @return true add success
     * @return false failed. Maybe because task has been added before
     */
    virtual bool add_task(task_id_t t_id) = 0;
    /**
	 * @brief Get the pointer to a task
	 * @param t_id Task id
	 * @return Pointer to the task
	 */
    virtual std::weak_ptr<Task> get_task(task_id_t t_id) = 0;
    /**
	 * @brief Add a dependence relationship that
	 * task \p who depends on task \p whom
	 */
    virtual void add_dependence(task_id_t who, task_id_t whom) = 0;
    /**
	 * @brief Block to wait a finished task. Executor won't keep any
	 * information about this task after wait() called
     * 
     * @return The pointer to the execution info
	 */
    virtual std::shared_ptr<ExecInfo> wait() = 0;
    /**
	 * @brief Block to wait a specific task
	 * @param task_id task id
	 * @return The pointer to the execution info
	 */
    virtual std::shared_ptr<ExecInfo> wait(task_id_t t_id) = 0;

protected:
    /**
     * @brief Notify executor a task has finished
     * @param t_id Id of task
     */
    virtual void notify_task_finish(task_id_t t_id) = 0;
};