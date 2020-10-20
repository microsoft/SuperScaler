project(superscaler_rt LANGUAGES CUDA CXX)

add_library(util OBJECT)
target_sources(
  util
  PUBLIC util.hpp
  PRIVATE util.cpp)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(util PUBLIC nlohmann_json::nlohmann_json)

add_subdirectory(executor)

# add_library(${PROJECT_NAME} SHARED)
add_library(${PROJECT_NAME} STATIC)

target_sources(
  ${PROJECT_NAME}
  PUBLIC session.hpp
  PRIVATE session.cpp)

target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC util executor)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
endif()

find_package(MPI 3.1 REQUIRED COMPONENTS C)
if(MPI_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_C)
endif()

enable_testing()
add_subdirectory(test)
