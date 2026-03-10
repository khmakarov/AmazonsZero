cmake_minimum_required(VERSION 3.28)
cmake_policy(SET CMP0148 NEW)  
project(Amazons LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)

pybind11_add_module(Amazons amazons.cpp game.cpp)

target_compile_options(Amazons PRIVATE -O3 -flto=auto)
target_include_directories(Amazons PRIVATE .)

set_target_properties(Amazons PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "Amazons"
)