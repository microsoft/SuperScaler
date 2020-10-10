find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(external_json)
endif()
