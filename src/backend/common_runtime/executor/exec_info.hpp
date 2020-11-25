// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <functional>

#include "task.hpp"

enum class ExecState { e_success, e_fail };

class ExecInfo {
public:
	ExecInfo();
	ExecInfo(task_id_t task_id, ExecState state);

	virtual ~ExecInfo();
	ExecState get_state() const;
	task_id_t get_task_id() const;

private:
	task_id_t m_id;
	ExecState m_state;
};
