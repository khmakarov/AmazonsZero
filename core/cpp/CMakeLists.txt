cmake_minimum_required(VERSION 3.12)
project(Amazons)

set(CMAKE_CXX_STANDARD 20)
find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(XXHASH REQUIRED IMPORTED_TARGET libxxhash)

pybind11_add_module(Amazons
    amazons.cpp
    action_index.cpp
    game_core.cpp
)

set_target_properties(Amazons PROPERTIES
    SUFFIX ".so"          
    PREFIX ""             
    OUTPUT_NAME "Amazons" 
)

target_include_directories(Amazons PRIVATE .)
target_link_libraries(Amazons PRIVATE PkgConfig::XXHASH)