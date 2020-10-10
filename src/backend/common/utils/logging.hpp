#pragma once

#include <limits>
#include <memory>
#include <sstream>

namespace superscaler
{
    enum
    {
        INFO = 0,
        WARNING = 1,
        ERROR = 2,
        NUM_SEVERITIES = 3
    };

    namespace log
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

    }; // namespace log

}; // namespace superscaler

#define _SC_LOG_INFO ::superscaler::log::Logger(__FILE__, __LINE__, ::superscaler::INFO).stream()
#define _SC_LOG_WARNING                                                                            \
    ::superscaler::log::Logger(__FILE__, __LINE__, ::superscaler::WARNING).stream()
#define _SC_LOG_ERROR ::superscaler::log::Logger(__FILE__, __LINE__, ::superscaler::ERROR).stream()

#define LOG(severity) _SC_LOG_##severity
