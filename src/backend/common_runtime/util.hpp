#pragma once

#include <sstream>
#include "nlohmann/json.hpp"

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
        class Logger
        {
        public:
            Logger(const char* fname, int line, int severity);
            std::ostream& stream() { return sstream_; }
            ~Logger();
            static int MinLogLevelFromEnv();

        protected:
            void GenerateLogMessage();

        private:
            const char* fname_;
            int line_;
            int severity_;
            std::stringstream sstream_;
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
