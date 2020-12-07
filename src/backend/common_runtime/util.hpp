// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include "nlohmann/json.hpp"

#define SC_DISALLOW_COPY_AND_ASSIGN(TypeName)                                                      \
    TypeName(const TypeName&) = delete;                                                            \
    void operator=(const TypeName&) = delete

namespace superscaler
{
    enum
    {
        INFO = 0,
        WARNING = 1,
        ERROR = 2,
        NUM_SEVERITIES = 3
    };

    namespace util
    {
        class NullLogger
        {
        public:
            NullLogger(const char* fname, int line, int severity)
                : fname_(fname)
                , line_(line)
                , severity_(severity)
            {
            }
            ~NullLogger() {}
            std::ostream& stream() { return sstream_; }

        protected:
            const char* fname_;
            int line_;
            int severity_;
            std::stringstream sstream_;
        };

        class Logger : public NullLogger
        {
        public:
            Logger(const char* fname, int line, int severity)
                : NullLogger(fname, line, severity)
            {
            }
            ~Logger();
            static int MinLogLevelFromEnv();
            static int MinVLogLevelFromEnv();
            static bool VlogActivated(const char*, int level);

        protected:
            void GenerateLogMessage();
        };

        using json = nlohmann::json;
        class JsonParser
        {
        public:
            static json load_from(std::string fpath);
        };

    }; // namespace util

}; // namespace superscaler

#define _SC_LOG_INFO ::superscaler::util::Logger(__FILE__, __LINE__, ::superscaler::INFO).stream()
#define _SC_LOG_WARNING                                                                            \
    ::superscaler::util::Logger(__FILE__, __LINE__, ::superscaler::WARNING).stream()
#define _SC_LOG_ERROR ::superscaler::util::Logger(__FILE__, __LINE__, ::superscaler::ERROR).stream()

#define LOG(severity) _SC_LOG_##severity

#define VLOG_IS_ON(lvl)                                                                            \
    (([](int level, const char* fname) -> bool {                                                   \
        static const bool vlog_activated =                                                         \
            ::superscaler::util::Logger::VlogActivated(fname, level);                              \
        return vlog_activated;                                                                     \
    })(lvl, __FILE__))

#define VLOG(level)                                                                                \
    (!VLOG_IS_ON(level))                                                                           \
        ? ::superscaler::util::NullLogger(__FILE__, __LINE__, ::superscaler::INFO).stream()        \
        : ::superscaler::util::Logger(__FILE__, __LINE__, ::superscaler::INFO).stream()
