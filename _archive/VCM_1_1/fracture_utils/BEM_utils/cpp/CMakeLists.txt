cmake_minimum_required(VERSION 3.12)
project(bem_solver)

set(CMAKE_CXX_STANDARD 14)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen3)

pybind11_add_module(bem_solver 
    bem_bindings.cpp
    bem_solver.cpp
)

set_target_properties(bem_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
