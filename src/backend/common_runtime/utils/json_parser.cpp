#include "json_parser.hpp"
#include <fstream>
#include <iostream>
namespace superscaler
{
    namespace json
    {
        json JsonParser::load_from(std::string fpath)
        {
            std::ifstream f(fpath);
            json j = json::parse(f);
            return j;
        }
    }; // namespace json

}; // namespace superscaler
