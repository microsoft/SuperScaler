// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <thread>
#include <sstream>
#include <unistd.h>

#include "utils.hpp"

std::string get_thread_unique_name(const std::string& prefix)
{
    std::ostringstream string_builder;
    string_builder << prefix << '_';
    string_builder << getpid() << '_';
    string_builder << std::this_thread::get_id();
    return string_builder.str();
}