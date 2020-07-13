#include <memory>
#include <unordered_map>

#include "task.hpp"

class PollExecutor;

class TaskManager {
public:
	TaskManager();
	TaskManager(const TaskManager &) = delete;
	TaskManager operator=(const TaskManager &) = delete;
	virtual ~TaskManager();

	/**
	 * @brief Create a task
	 * @return A task id unique within the process
	 */
	template <class T, class... Args>
	task_id_t create_task(Args... args);

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
	std::shared_ptr<Task> get_task(task_id_t t_id);

private:
	// all created tasks stored here
	std::unordered_map<task_id_t, std::shared_ptr<Task> > m_tasks;
	task_id_t m_task_count;
};
