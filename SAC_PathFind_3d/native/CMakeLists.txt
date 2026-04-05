cmake_minimum_required(VERSION 3.16)

project(sac_pathfind_detour_wrapper LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

add_library(detour_navmesh_wrapper STATIC
    detour_navmesh_wrapper.cpp
)

target_include_directories(detour_navmesh_wrapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(detour_navmesh_wrapper PUBLIC cxx_std_17)

add_library(detour_navmesh_py MODULE
    python_module.cpp
)

target_include_directories(detour_navmesh_py
    PRIVATE
        ${Python3_INCLUDE_DIRS}
)

target_compile_features(detour_navmesh_py PUBLIC cxx_std_17)
target_link_libraries(detour_navmesh_py PRIVATE detour_navmesh_wrapper)
set_target_properties(detour_navmesh_py PROPERTIES PREFIX "")

# Required external dependencies:
#   1. RecastNavigation / Detour headers and library
#   2. pybind11 headers
#
# Example configuration:
#   target_include_directories(detour_navmesh_wrapper PRIVATE
#       /path/to/recastnavigation/Detour/Include
#       /path/to/pybind11/include
#   )
#   target_include_directories(detour_navmesh_py PRIVATE
#       /path/to/pybind11/include
#   )
#   target_link_libraries(detour_navmesh_wrapper PRIVATE Detour)
