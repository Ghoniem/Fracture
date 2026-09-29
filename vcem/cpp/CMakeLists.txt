cmake_minimum_required(VERSION 3.18)
project(vcem_4_0_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Eigen (vendored sibling Libraries/, same layout as RadCluster) ────────────
get_filename_component(LIBRARIES_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../Libraries" ABSOLUTE)
set(EIGEN_INCLUDE_DIR "${LIBRARIES_DIR}/eigen-3.4.0")
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR
        "Eigen headers not found at ${EIGEN_INCLUDE_DIR}.\n"
        "Expected D:/GitHub/Libraries/eigen-3.4.0 (sibling of repo root).")
endif()
message(STATUS "Eigen: ${EIGEN_INCLUDE_DIR}")

# ── Python (active interpreter -> conda vcem_4_0 env) ─────────────────────────
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python: ${Python_EXECUTABLE} (v${Python_VERSION})")

# ── pybind11 (from the active Python env) ─────────────────────────────────────
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pb_rc)
if(NOT _pb_rc EQUAL 0)
    message(FATAL_ERROR
        "pybind11 not importable from ${Python_EXECUTABLE}.\n"
        "Install with:  ${Python_EXECUTABLE} -m pip install pybind11")
endif()
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11: ${pybind11_DIR}")

# ── Compiler optimization (same recipe as RadCluster_2_0) ─────────────────────
if(MSVC)
    add_compile_options(/arch:AVX2 /GL /fp:precise /W3 /EHsc)
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ── OpenMP (parallelizes BEM assembly and stress_on_grid) ─────────────────────
# Falls back to a single-threaded build if OpenMP is unavailable. Code paths
# guarded by VCEM_HAVE_OPENMP=1 take the parallel-for branch; otherwise the
# serial loop runs unchanged.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found -- parallel BEM ENABLED")
else()
    message(WARNING "OpenMP not found -- BEM will run single-threaded")
endif()

# ── Core BEM library (static, header + impl in src/, headers in include/) ─────
add_library(bem_core STATIC
    src/quadrature.cpp
    src/kelvin.cpp
    src/bem_solver.cpp
    src/edge_dislocation.cpp
    src/kkt.cpp
    src/crack_assemblers.cpp
    src/edge_intersection.cpp
)
target_include_directories(bem_core PUBLIC
    ${EIGEN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bem_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(bem_core PUBLIC VCEM_HAVE_OPENMP=1)
endif()

# ── pybind11 extension module (this is what notebooks import) ─────────────────
pybind11_add_module(bem_cpp MODULE
    bindings/bem_cpp_module.cpp
)
target_link_libraries(bem_cpp PRIVATE bem_core)
target_include_directories(bem_cpp PRIVATE
    ${EIGEN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Install the .pyd next to the python helper package ────────────────────────
# After `cmake --build . --config Release --target install`, bem_cpp.*.pyd
# lands in vcem/cpp/python/bem_cpp/ so it can be imported as
# `from bem_cpp import bem_cpp` (or added to sys.path).
install(TARGETS bem_cpp
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/bem_cpp"
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/bem_cpp"
)
