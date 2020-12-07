// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "util.hpp"

#include <chrono>
#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>

namespace superscaler
{
    namespace util
    {
        //Logger
        int Logger::MinLogLevelFromEnv()
        {
            const char* sc_env_var_val = getenv("SC_MIN_LOG_LEVEL");
            if (sc_env_var_val)
                return std::stoi(sc_env_var_val);
            else
                return 0;
        }

        int Logger::MinVLogLevelFromEnv()
        {
            const char* sc_env_var_val = getenv("SC_MIN_VLOG_LEVEL");
            if (sc_env_var_val)
                return std::stoi(sc_env_var_val);
            else
                return 0;
        }

        bool Logger::VlogActivated(const char* fname, int level)
        {
            return (level <= MinVLogLevelFromEnv());
        }

        Logger::~Logger()
        {
            if (severity_ >= Logger::MinLogLevelFromEnv())
                GenerateLogMessage();
        }
        void Logger::GenerateLogMessage()
        {
            uint64_t now_micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::high_resolution_clock::now().time_since_epoch())
                                      .count();
            time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
            int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
            const size_t time_buffer_size = 30;
            char time_buffer[time_buffer_size];
            strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S", localtime(&now_seconds));
            std::string filename = std::string(fname_);
            std::string partial_name = filename.substr(filename.rfind("src"));
            std::fprintf(stderr,
                         "%s.%06d: %c %s:%d] %s\n",
                         time_buffer,
                         micros_remainder,
                         ("IWE"[severity_]),
                         partial_name.c_str(),
                         line_,
                         sstream_.str().c_str());
        }

        //Json Parser
        json JsonParser::load_from(std::string fpath)
        {
            std::ifstream f(fpath);
            json j = json::parse(f);
            return j;
        }
    } // namespace util
};    // namespace superscaler