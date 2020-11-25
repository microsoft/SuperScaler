// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "task_manager.hpp"

TaskManager::TaskManager(task_id_t max_task_id)
	: m_cur_max_id(0), m_max_task_id(max_task_id)
{
}

TaskManager::~TaskManager()

{
}

bool TaskManager::delete_task(task_id_t t_id)
{
	if (m_tasks.erase(t_id) > 0) {
		m_recycle_queue.push(t_id);
		return true;
	} else
		return false;
}

std::shared_ptr<Task> TaskManager::get_task(task_id_t t_id) const
{
	auto itr = m_tasks.find(t_id);
	if (itr != m_tasks.end())
		return itr->second;
	else
		return nullptr;
}

size_t TaskManager::get_active_task_num() const
{
	return m_cur_max_id - m_recycle_queue.size();
}
