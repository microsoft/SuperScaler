#pragma once

#include "nlohmann/json.hpp"

namespace superscaler
{
    namespace json
    {
        using json = nlohmann::json;
        class JsonParser
        {
        public:
            static json load_from(std::string fpath);
        };
    }; // namespace json

}; // namespace superscaler
