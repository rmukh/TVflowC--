cmake_minimum_required (VERSION 3.8)
project (tv_flow_python)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /nologo /Gm- /W3 /WX /EHsc /GS /fp:fast /permissive- /Zc:wchar_t /Zc:forScope /Zc:inline /Gd /TP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FC /wd4996")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /RTC1 /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Gy- /MD")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /OPT:NOICF")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG /OPT:NOICF")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -march=native -pthread -flto -ffast-math")
endif()

if (NOT DEFINED Eigen3_DIR)
    set(CMAKE_PREFIX_PATH "D:\\Eigen3" PATH)
else()
    message(STATUS "Eigen3_DIR is defined in " ${Eigen3_DIR})
endif()
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
link_directories(${Eigen_INCLUDE_DIRS})

# Make a library of the TVflow code
add_library(TVflow_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/tvflow.cpp)

# Make a python module with an pybind11 interface
add_subdirectory(pybind11)
pybind11_add_module(tv_flow_python ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/tvflow.cpp ${CMAKE_CURRENT_SOURCE_DIR}/TVflow/python_wrapper.cpp)
target_link_libraries(tv_flow_python PUBLIC Eigen3::Eigen)

# Make an executable and link it to the library
add_executable(main main.cpp)
target_link_libraries(TVflow_lib Eigen3::Eigen)
target_link_libraries(main TVflow_lib)
