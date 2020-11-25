// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "exec_info.hpp"

ExecInfo::ExecInfo()
	: m_id(0)
{
}

ExecInfo::ExecInfo(task_id_t task_id, ExecState state)
	: m_id(task_id), m_state(state)
{
}

ExecInfo::~ExecInfo()
{
}

ExecState ExecInfo::get_state() const
{
	return m_state;
}

task_id_t ExecInfo::get_task_id() const
{
	return m_id;
}
