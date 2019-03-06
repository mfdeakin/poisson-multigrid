
cmake_minimum_required (VERSION 3.3)
project(multigrid)

enable_testing()

include_directories(./include ./xtensor/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_library(multigrid_poisson ./src/multigrid.cpp)
target_compile_options(multigrid_poisson PUBLIC -fPIC)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(multigrid ./src/bindpy.cpp)
target_link_libraries(multigrid PRIVATE multigrid_poisson)
set_target_properties(multigrid PROPERTIES
                      CXX_STANDARD 17
		      CXX_STANDARD_REQUIRED ON
		      CXX_EXTENSIONS OFF)

add_executable(multigrid_tester ./src/test.cpp)
target_link_libraries(multigrid_tester multigrid_poisson gtest)
add_test(multigrid_test multigrid_tester)

configure_file(./scripts/v-cycle.py v-cycle.py COPYONLY)
