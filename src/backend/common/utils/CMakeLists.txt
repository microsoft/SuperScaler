add_library(logger OBJECT)
target_sources(
  logger
  PUBLIC logging.hpp
  PRIVATE logging.cpp)
target_include_directories(logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(json_parser OBJECT)
target_sources(
  json_parser
  PUBLIC json_parser.hpp
  PRIVATE json_parser.cpp)
target_include_directories(json_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(json_parser PUBLIC nlohmann_json::nlohmann_json)
