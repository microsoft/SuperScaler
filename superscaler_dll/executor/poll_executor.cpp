#include "poll_executor.hpp"

PollExecutor::PollExecutor()
{
}

PollExecutor::~PollExecutor()
{
}

bool PollExecutor::add_task(task_id_t)
{
	return true;
}

std::weak_ptr<Task> PollExecutor::get_task(task_id_t)
{
	return std::make_shared<Task>(nullptr, nullptr);
}

void PollExecutor::add_dependence(task_id_t, task_id_t)
{
}

std::shared_ptr<ExecInfo> PollExecutor::wait()
{
	return nullptr;
}

std::shared_ptr<ExecInfo> PollExecutor::wait(task_id_t)
{
	return nullptr;
}

void PollExecutor::notify_task_finish(task_id_t)
{
}

