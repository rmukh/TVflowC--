cmake_minimum_required (VERSION 3.8)
project (TVflow)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3 -pthread -fno-math-errno -mfma -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -pthread -fno-math-errno -mfma -ffast-math")

if (NOT DEFINED Eigen3_DIR)
    set(CMAKE_PREFIX_PATH "D:\\Eigen3" PATH)
else()
    message(STATUS "Eigen3_DIR is defined in " ${Eigen3_DIR})
endif()
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
link_directories(${Eigen_INCLUDE_DIRS})

# Make a static library of the TVflow code
add_library(TVflow_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/tvflow.cpp)

# Make a python module with an pybind11 interface
if (NOT DEFINED pybind11_DIR)
    set(CMAKE_PREFIX_PATH "D:\\pybind11\\share\\cmake\\pybind11" PATH)
else()
    message(STATUS "pybind11_DIR is defined in " ${pybind11_DIR})
endif()

find_package(pybind11 REQUIRED)
pybind11_add_module(tv_flow_python ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/TVflow.cpp ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/python_wrapper.cpp)
target_link_libraries(tv_flow_python PUBLIC Eigen3::Eigen m)

# Make an executable and link it to the static library
add_executable(main main.cpp)
target_link_libraries(TVflow_lib Eigen3::Eigen)
target_link_libraries(main TVflow_lib)
