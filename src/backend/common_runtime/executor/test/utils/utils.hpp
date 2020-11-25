// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <iostream>

/**
* \brief Get thread unique name like: prefix_pid_tid
* 
* @param prefix 
* @return std::string 
**/
std::string get_thread_unique_name(const std::string &prefix);