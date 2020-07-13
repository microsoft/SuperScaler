#pragma once

#include <memory>
#include <functional>

#include "task.hpp"

enum class ExecState { e_success, e_fail };

class ExecInfo {
public:
	ExecInfo() = delete;
	ExecInfo(const ExecInfo &) = delete;
	ExecInfo &operator=(const ExecInfo &) = delete;

	ExecInfo(task_id_t task_id ,task_callback_t callback);

	virtual ~ExecInfo();
	ExecState get_state() const;

private:
	task_id_t m_id;
	ExecState m_state;
	task_callback_t m_callback;
};
