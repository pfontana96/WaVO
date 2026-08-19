cmake_minimum_required(VERSION 3.26)

project(
  wavo
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Everything ends up inside a Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Keep the conan-cache library dirs (libtbb.so.12 from the Open3D package) in
# the installed extension's RUNPATH; CMake strips it on install otherwise and
# _core then only imports if something else (e.g. the open3d wheel) already
# loaded a libtbb into the process. Dev-build tradeoff: bakes absolute conan
# paths into the .so, so a redistributable wheel would need bundling instead.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(pybind11 CONFIG REQUIRED)
# Provided by Conan (conanfile.txt) when building through pip, or by any
# system/toolchain OpenCV ≥ 4 when configuring the C++ library directly.
find_package(OpenCV 4 REQUIRED)
# Conan builds this from the in-repo recipe (conan/open3d); a system
# Open3D works too when configuring the C++ library directly. No version
# arg: Conan pins it, and Open3D's config only accepts exact matches.
find_package(Open3D REQUIRED)
# Only used inside src/Parameters.cpp; not part of the public API.
find_package(yaml-cpp REQUIRED)

# Core C++ library — no Python dependency, so it can be reused / tested natively.
file(GLOB_RECURSE WAVO_CORE_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(wavo_core STATIC ${WAVO_CORE_SOURCES})
target_include_directories(
  wavo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
                   ${CMAKE_CURRENT_SOURCE_DIR}/include)
# PUBLIC: cv::Mat appears in public headers (RGBDFrame.hpp); Open3D types
# will appear in the 3D-registration API the same way.
target_link_libraries(wavo_core PUBLIC opencv_core opencv_imgproc
                                       opencv_calib3d Open3D::Open3D)
target_link_libraries(wavo_core PRIVATE yaml-cpp::yaml-cpp)

# Python extension module (wavo._core)
pybind11_add_module(_core bindings/wavo_bindings.cpp)
target_link_libraries(_core PRIVATE wavo_core)
target_compile_definitions(_core PRIVATE WAVO_VERSION_INFO="${PROJECT_VERSION}")

install(TARGETS _core LIBRARY DESTINATION wavo)
