cmake_minimum_required(VERSION 3.26)

project(
  wavo
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# Core C++ library — no Python dependency, so it can be reused / tested natively.
add_library(wavo_core STATIC cpp/src/core.cpp)
target_include_directories(wavo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
set_target_properties(wavo_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python extension module (wavo._core)
pybind11_add_module(_core bindings/wavo_bindings.cpp)
target_link_libraries(_core PRIVATE wavo_core)
target_compile_definitions(_core PRIVATE WAVO_VERSION_INFO="${PROJECT_VERSION}")

install(TARGETS _core LIBRARY DESTINATION wavo)
